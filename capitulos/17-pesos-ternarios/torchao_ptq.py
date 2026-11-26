"""TorchAO INT8 weight-only PTQ on the very same trained FP32 MLP."""
import copy
import io
import json

import torch
import torchao
from torch import nn
from torchao.quantization import Int8WeightOnlyConfig, quantize_

from lab_ternario import build, dataset, evaluate


def serialized_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    # Train the chapter MLP in FP32 with its original minibatch order.
    original = build("mlp", "fp32")
    opt = torch.optim.AdamW(original.parameters(), lr=0.003)
    x, y = dataset(1536, 2026)
    g = torch.Generator().manual_seed(10007)
    for _ in range(8):
        for ids in torch.randperm(len(y), generator=g).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(original(x[ids]), y[ids]).backward()
            opt.step()

    # Export to plain nn.Linear so the example only uses standard module semantics.
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 64, bias=False), nn.ReLU(),
                        nn.Linear(64, 32, bias=False), nn.ReLU(), nn.Linear(32, 4, bias=False))
    mlp.load_state_dict(original.state_dict())
    mlp.eval()
    mlp_int8 = copy.deepcopy(mlp)
    quantize_(mlp_int8, Int8WeightOnlyConfig(set_inductor_config=False))

    with torch.no_grad():
        test = dataset(512, 2027)
        assert torch.equal(original(test[0]), mlp(test[0]))
        logits, logits_int8 = mlp(test[0]), mlp_int8(test[0])
        first = mlp_int8[1].weight
        assert first.qdata.dtype == torch.int8
        assert all(m.weight.qdata.dtype == torch.int8 for m in mlp_int8 if isinstance(m, nn.Linear))
        print(json.dumps({"torch": torch.__version__, "torchao": torchao.__version__,
                          "config": "Int8WeightOnlyConfig", "model_seed": 7,
                          "fp32": evaluate(mlp, test), "int8": evaluate(mlp_int8, test),
                          "max_abs_logit_error": float((logits_int8 - logits).abs().max()),
                          "state_dict_bytes_fp32": serialized_bytes(mlp),
                          "state_dict_bytes_int8": serialized_bytes(mlp_int8),
                          "weight_type": type(first).__name__,
                          "underlying_qdata_dtype": str(first.qdata.dtype),
                          "scale_shape_first": list(first.scale.shape)}, indent=2))


if __name__ == "__main__":
    main()
