"""Rebuild the chapter's PNGs, using Pillow and DejaVu Sans."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BG = "#101C30"
TXT = "#F4FAFF"
MUTED = "#ABBFE1"
COLORS = ["#75B6FF", "#FFAE72", "#72E1C2"]
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / family), size)
    except OSError:
        return ImageFont.truetype(family, size)


def label(draw, pos, msg, size=28, color=TXT, bold=False, anchor=None):
    draw.text(pos, msg, fill=color, font=font(size, bold), anchor=anchor)


def canvas(number, title, subtitle):
    im = Image.new("RGB", (1400, 850), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1400, 9), fill=COLORS[2])
    label(d, (70, 50), f"PYTORCH NA PRÁTICA   ·   CAPÍTULO 17   /   {number:02}", 22, COLORS[2], True)
    label(d, (70, 94), title, 54, TXT, True)
    label(d, (72, 174), subtitle, 29, MUTED)
    return im, d


def cover(out):
    im, d = canvas(1, "E se o peso só puder ser −1, 0 ou +1?",
                   "Os mesmos três códigos. Duas formas de usá-los no cálculo.")
    for x, w, color, big, desc in [(72, "−1", COLORS[0], "Inverte o sinal", "contribui com −x"),
                                   (500, "0", MUTED, "Ignora a entrada", "contribui com 0"),
                                   (928, "+1", COLORS[2], "Preserva o sinal", "contribui com +x")]:
        d.rounded_rectangle((x, 260, x + 400, 554), radius=30, fill="#192B47", outline="#344C72", width=2)
        label(d, (x + 200, 310), w, 94, color, True, "mt")
        label(d, (x + 200, 440), big, 33, TXT, True, "mt")
        label(d, (x + 200, 494), desc, 27, MUTED, False, "mt")
    d.line((72, 640, 1328, 640), fill="#344C72", width=3)
    label(d, (72, 684), "Estrito:  q = [−1, 0, +1]", 37, COLORS[1], True)
    label(d, (72, 742), "Com escala α = 0,25:  αq = [−0,25, 0, +0,25]", 37, COLORS[2], True)
    im.save(out / "01-ternario.png")


def bars(out, rows):
    im, d = canvas(2, "O que mudou quando tiramos a escala?",
                   "Acurácia no teste · 3 sementes · barras de erro: ±1 desvio padrão amostral")
    for i, s in enumerate(["FP32", "Ternário estrito", "Ternário com escala"]):
        x = (85, 357, 735)[i]
        d.rounded_rectangle((x, 246, x + 26, 272), 4, fill=COLORS[i])
        label(d, (x + 39, 245), s, 24, TXT, True)
    left, width = 295, 990
    for tick in (0, 25, 50, 75, 100):
        x = left + width * tick / 100
        d.line((x, 327, x, 739), fill=("#BBC7DD" if tick == 25 else "#324562"),
               width=(3 if tick == 25 else 1))
        label(d, (x, 750), str(tick) + "%", 20, MUTED, False, "mt")
    label(d, (left + width / 4 + 10, 296), "25%: referência ao acaso", 20, MUTED)
    for i, arch in enumerate(("mlp", "cnn", "transformer")):
        top = 339 + 137 * i
        label(d, (81, top + 35), {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer"}[arch],
              25, TXT, True)
        for j, mode in enumerate(("fp32", "strict", "scaled")):
            r = next(r for r in rows if r["arch"] == arch and r["mode"] == mode)
            value, sd = (r[f"final_accuracy_pct_{stat}"] for stat in ("mean", "sd"))
            y = top + 39 * j
            end = left + width * value / 100
            d.rounded_rectangle((left, y, end, y + 33), 5, fill=COLORS[j])
            lo, hi = left + width * (value - sd) / 100, left + width * (value + sd) / 100
            d.line((lo, y + 17, hi, y + 17), fill=TXT, width=2)
            for x in (lo, hi):
                d.line((x, y + 10, x, y + 24), fill=TXT, width=2)
            # Place number inside bar away from error whiskers for legibility.
            label(d, (left + 12, y + 6), f"{value:.2f}%".replace(".", ","), 19, BG, True)
    label(d, (82, 800), "Dados geométricos sintéticos · 512 imagens de teste · CPU · não é benchmark de latência", 22, MUTED)
    im.save(out / "02-resultados.png")


def flowing(out):
    im, d = canvas(3, "Do treino à inferência",
                   "Ternário didático e TorchAO INT8 resolvem partes diferentes do problema.")
    for y, name, color, steps in [
        (265, "Ternário didático", COLORS[1], ["Pesos mestres FP32", "Códigos −1, 0, +1", "Forward em float"]),
        (510, "TorchAO · PTQ", COLORS[2], ["Treine em FP32", "quantize_ em cópia", "Pesos INT8 + escalas"])]:
        label(d, (72, y), name, 34, color, True)
        for i, msg in enumerate(steps):
            x = 72 + i * 446
            d.rounded_rectangle((x, y + 63, x + 360, y + 157), 20, fill="#192B47", outline="#344C72", width=2)
            label(d, (x + 180, y + 110), msg, 25, TXT, True, "mm")
            if i < 2:
                label(d, (x + 402, y + 110), "→", 38, color, True, "mm")
    label(d, (72, 747), "STE permite retropropagar no treino ternário. PTQ faz a conversão só após o treino.", 26, MUTED)
    im.save(out / "03-fluxos.png")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    out = base / "figuras"
    out.mkdir(exist_ok=True)
    cover(out)
    bars(out, json.loads((base / "resultados.json").read_text())["summary"])
    flowing(out)
