"""Teste de integração em loopback com quatro servidores locais independentes."""
from contextlib import ExitStack
import os
import json
from pathlib import Path
import selectors
import subprocess
import sys
import unittest

import torch

from infer_client import ReplicaClient
from serve_ddp import request
from train_ddp import N_FEATURES, model

HERE = Path(__file__).resolve().parent


def launch(stack, checkpoint, role, identity, peer=None):
    cmd = [sys.executable, str(HERE / "serve_ddp.py"), "--checkpoint",
           checkpoint, "--role", role, "--id", identity, "--port", "0"]
    if peer:
        cmd += ["--peer", peer]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    stack.callback(stop, p)
    with selectors.DefaultSelector() as selector:
        selector.register(p.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=15):
            raise RuntimeError(f"{identity} não iniciou em 15 s")
        address = p.stdout.readline().strip().rsplit(" ", 1)[-1]
    if not address.startswith("127.0.0.1:"):
        raise RuntimeError(f"{identity} não publicou endereço de loopback")
    return p, "http://" + address


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    process.stdout.close()


@unittest.skipUnless(os.getenv("DDP_CHECKPOINT"), "gere ou forneça um checkpoint do treino")
class InferenceTests(unittest.TestCase):
    def test_replicas_round_robin_failover_e_cadeia(self):
        checkpoint = str(Path(os.environ["DDP_CHECKPOINT"]).resolve())
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"]
        net = model().eval()
        net.load_state_dict(weights)
        values = [i / N_FEATURES for i in range(N_FEATURES)]
        with torch.inference_mode():
            expected = net(torch.tensor([values], dtype=torch.float32))[0]
        with ExitStack() as stack:
            a, a_url = launch(stack, checkpoint, "replica", "R0")
            _, b_url = launch(stack, checkpoint, "replica", "R1")
            tail, tail_url = launch(stack, checkpoint, "stage-b", "B")
            _, head_url = launch(stack, checkpoint, "stage-a", "A", tail_url)
            for url in (a_url, b_url, head_url, tail_url):
                self.assertEqual(request(url, "/ready")[0], 200)

            client = ReplicaClient([a_url, b_url])
            seen = {"R0": 0, "R1": 0}
            for _ in range(12):
                out = client.predict(values)
                seen[out["id"]] += 1
                torch.testing.assert_close(torch.tensor(out["logits"]), expected, rtol=0, atol=0)
                self.assertEqual(out["class"], int(expected.argmax()))
            self.assertEqual(seen, {"R0": 6, "R1": 6})
            self.assertEqual(client.next_index, 0)
            cli = subprocess.run(
                [sys.executable, str(HERE / "infer_client.py"),
                 "--endpoint", a_url, "--endpoint", b_url, "--requests", "2"],
                check=True, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual([json.loads(line)["id"] for line in cli.stdout.splitlines()],
                             ["R0", "R1"])
            stop(a)
            self.assertEqual(client.predict(values)["id"], "R1")  # A primeira tentativa foi à R0.
            self.assertEqual(request(b_url, "/predict", {"values": []})[0], 400)
            print(f"replicas={seen} failover_R0_para_R1=ok rejeicao_entrada=400", flush=True)

            status, output = request(head_url, "/predict", {"values": values}, timeout=5)
            self.assertEqual(status, 200)
            logits = torch.tensor(output["logits"])
            torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(output["class"], int(expected.argmax()))
            gap = (logits - expected).abs().max().item()
            stop(tail)
            self.assertEqual(request(head_url, "/ready", timeout=5)[0], 503)
            self.assertEqual(request(head_url, "/predict", {"values": values}, timeout=5)[0], 502)
            print(f"split_vs_local_max_abs_diff={gap:.3e} split_sem_B_ready=503 predict=502", flush=True)


if __name__ == "__main__":
    unittest.main()
