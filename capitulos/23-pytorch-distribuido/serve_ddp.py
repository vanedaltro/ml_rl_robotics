"""Inferência HTTP didática. Somente loopback ou rede privada confiável."""
import argparse
import http.client
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import torch

from train_ddp import N_CLASSES, N_FEATURES, model

LIMIT = 16384
TIMEOUT = 2
HIDDEN = 64


def endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http" or not parsed.hostname or parsed.username
        or parsed.password or parsed.path not in ("", "/")
        or parsed.query or parsed.fragment
    ):
        raise ValueError("use http://host:porta, sem caminho")
    # Force validation up front for an invalid port.
    _ = parsed.port
    return parsed


def request(target, path, payload=None, timeout=TIMEOUT):
    """As URLs dos nós são configuração fixa, não fornecidas por pedidos HTTP."""
    addr = endpoint(target)
    conn = http.client.HTTPConnection(addr.hostname, addr.port, timeout=timeout)
    try:
        data = None if payload is None else json.dumps(payload, allow_nan=False)
        conn.request("GET" if payload is None else "POST", path, data,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("resposta HTTP grande demais")
        return response.status, json.loads(raw)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise ConnectionError(f"{target}: {exc}") from exc
    finally:
        conn.close()


def vector(payload, width):
    if not isinstance(payload, dict):
        raise ValueError("esperado um objeto JSON")
    values = payload.get("values")
    if not isinstance(values, list) or len(values) != width:
        raise ValueError(f"values deve conter exatamente {width} números")
    if any(type(x) not in (int, float) or not (-1e6 <= x <= 1e6) for x in values):
        raise ValueError("use apenas números finitos com módulo <= 1e6")
    return torch.tensor([values], dtype=torch.float32)


def make_handler(net, role, identity, peer):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(TIMEOUT)

        def log_message(self, *_args):
            pass

        def reply(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/ready":
                return self.reply(404, {"error": "rota não encontrada"})
            if role == "stage-a":
                try:
                    status, body = request(peer, "/ready")
                    if status != 200 or body.get("role") != "stage-b":
                        raise ConnectionError("stage-b indisponível")
                except (ConnectionError, AttributeError):
                    return self.reply(503, {"ready": False, "role": role, "id": identity})
            self.reply(200, {"ready": True, "role": role, "id": identity})

        def do_POST(self):
            route = "/stage" if role == "stage-b" else "/predict"
            if self.path != route:
                return self.reply(404, {"error": "rota não encontrada"})

            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= LIMIT:
                    raise ValueError("corpo ausente ou grande demais")
                payload = json.loads(self.rfile.read(n))
                x = vector(payload, HIDDEN if role == "stage-b" else N_FEATURES)
            except (ValueError, OSError) as exc:
                return self.reply(400, {"error": str(exc)})

            with torch.inference_mode():
                if role == "stage-a":
                    hidden = net(x)[0].tolist()
                    try:
                        status, body = request(peer, "/stage", {"values": hidden})
                        if status != 200 or not isinstance(body, dict):
                            raise ConnectionError("stage-b indisponível")
                        values = body.get("logits")
                        if (not isinstance(values, list) or len(values) != N_CLASSES
                                or any(type(v) not in (float, int) or not math.isfinite(v)
                                       for v in values)):
                            raise ConnectionError("logits inválidos")
                    except ConnectionError:
                        return self.reply(502, {"error": "stage-b indisponível"})
                    body["id"] = identity
                    return self.reply(200, body)
                logits = net(x)[0]
                self.reply(200, {"logits": logits.tolist(), "class": int(logits.argmax()),
                                 "id": identity})
    return Handler


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--role", choices=("replica", "stage-a", "stage-b"), default="replica")
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--id", default="local")
    cli.add_argument("--peer", help="URL HTTP base do stage-b (obrigatória só no stage-a)")
    args = cli.parse_args()
    if (args.role == "stage-a") != bool(args.peer):
        cli.error("--peer deve ser usada somente no stage-a")
    if args.peer:
        try:
            endpoint(args.peer)
        except ValueError as exc:
            cli.error(str(exc))
    torch.set_num_threads(1)
    # O servidor aceita somente checkpoint local confiável; não recebe arquivos pela rede.
    full = model()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    full.load_state_dict(saved["model_state"])
    full.eval()
    net = {"replica": lambda: full, "stage-a": lambda: full[:2],
           "stage-b": lambda: full[2:]}[args.role]()
    del full, saved
    with ThreadingHTTPServer((args.host, args.port), make_handler(
            net, args.role, args.id, args.peer)) as server:
        print(f"{args.role} {args.id} escutando em {args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
