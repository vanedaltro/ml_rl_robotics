"""Demonstração separada: operação Linear dinâmica real em CPU (API legada)."""
import io
import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic


def tamanho(modelo):
    buf = io.BytesIO()
    torch.save(modelo.state_dict(), buf)
    return len(buf.getbuffer())


def main():
    engines = torch.backends.quantized.supported_engines
    engine = "qnnpack" if "qnnpack" in engines else next(
        (name for name in ("x86", "fbgemm", "onednn") if name in engines), None
    )
    if engine is None:
        raise RuntimeError(f"Esta build não tem backend INT8 para o exemplo: {engines}")
    torch.backends.quantized.engine = engine
    torch.set_num_threads(1)
    torch.manual_seed(7)
    fp32 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4)).eval()
    int8 = quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8, inplace=False)
    x = torch.rand(8, 256)
    with torch.inference_mode():
        y, y_int8 = fp32(x), int8(x)
    assert y.shape == y_int8.shape and int8[0].weight().dtype == torch.qint8
    print("Torch:", torch.__version__, "| backend:", engine)
    print("Tipo de peso:", int8[0].weight().dtype)
    print("Serialização do state_dict:", tamanho(fp32), "→", tamanho(int8), "bytes")
    print("Erro absoluto máximo nesse lote:", float((y - y_int8).abs().max()))


if __name__ == "__main__":
    main()
