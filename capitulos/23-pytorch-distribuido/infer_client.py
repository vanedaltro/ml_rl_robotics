"""Cliente didático: round robin entre réplicas e nova tentativa em caso de falha."""
import argparse
import json
from dataclasses import dataclass

from serve_ddp import endpoint, request
from train_ddp import N_FEATURES


@dataclass
class ReplicaClient:
    endpoints: list[str]
    timeout: float = 2
    next_index: int = 0

    def __post_init__(self):
        if not self.endpoints:
            raise ValueError("configure pelo menos um endpoint")

    def predict(self, values):
        start = self.next_index
        self.next_index = (start + 1) % len(self.endpoints)
        for i in range(len(self.endpoints)):
            target = self.endpoints[(start + i) % len(self.endpoints)]
            try:
                status, result = request(target, "/predict", {"values": values}, self.timeout)
            except ConnectionError:
                continue
            if status >= 500:
                continue
            if status != 200:
                raise ValueError(f"HTTP {status}: {result}")
            return result
        raise ConnectionError("nenhuma réplica respondeu à inferência")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--endpoint", action="append", required=True)
    cli.add_argument("--requests", type=int, default=1)
    args = cli.parse_args()
    try:
        for url in args.endpoint:
            endpoint(url)
        if args.requests < 1:
            raise ValueError("--requests precisa ser positivo")
        client = ReplicaClient(args.endpoint)
        for _ in range(args.requests):
            output = client.predict([i / N_FEATURES for i in range(N_FEATURES)])
            print(json.dumps(output, ensure_ascii=False), flush=True)
    except (ValueError, ConnectionError) as exc:
        cli.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
