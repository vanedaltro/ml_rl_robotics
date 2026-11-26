"""Gera figuras originais em Pillow; inclui fontes padrão do macOS."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONTS = Path("/System/Library/Fonts/Supplemental")
W, H = 1440, 876
BG, INK, MUTED = "#F5F8FC", "#10283C", "#405A70"
TEAL, TEAL_BG, WHITE, LINE = "#006D63", "#DFF7F2", "#FFFFFF", "#D4E2EE"
AMBER, AMBER_BG = "#914715", "#FFF1DD"
BLUE, BLUE_BG = "#215C9C", "#E5F1FF"


def font(size, bold=False):
    return ImageFont.truetype(str(FONTS / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)


def label(d, x, y, content, size=26, color=INK, bold=False, anchor="lt"):
    f = font(size, bold)
    d.text((x, y), content, fill=color, font=f, anchor=anchor)


def rect(d, x0, y0, x1, y1, fill=WHITE, border=None, radius=23):
    d.rounded_rectangle((x0, y0, x1, y1), radius, fill, outline=border, width=2 if border else 1)


def arrow(d, x0, y, x1, c=TEAL):
    d.line((x0, y, x1 - 15, y), fill=c, width=6)
    d.polygon(((x1 - 18, y - 11), (x1, y), (x1 - 18, y + 11)), fill=c)


def base(seq, title, subtitle):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 15), fill=TEAL)
    rect(d, 72, 55, 465, 99, TEAL_BG, radius=20)
    label(d, 91, 68, f"CAPÍTULO 21    /    DIAGRAMA {seq}", 22, TEAL, True)
    label(d, 72, 128, title, 63, INK, True)
    label(d, 75, 220, subtitle, 31, MUTED)
    d.line((72, 807, 1368, 807), fill=LINE, width=2)
    label(d, 72, 826, "VÂNIA DALTRO  /  SÉRIE ML, RL E ROBÓTICA", 22, MUTED, True)
    label(d, 1368, 827, "PYTORCH   ·   SIMULAÇÃO", 21, MUTED, True, "rt")
    return im, d


def save(im, name):
    im.save(ROOT / name, optimize=True)


def track():
    im, d = base("01", "Uma pista que muda de regra", "O passo comum para, mas a tração ainda leva à próxima casa.")
    for i in range(9):
        x = 74 + 145 * i
        problem = i in {3, 5}
        goal = i == 8
        color, fill = (AMBER, AMBER_BG) if problem else (TEAL, TEAL_BG) if goal else (INK, WHITE)
        rect(d, x, 293, x + 132, 426, fill, LINE)
        label(d, x + 66, 323, str(i), 54, color, True, "mt")
        label(d, x + 66, 386, "MOLHADO" if problem else "META" if goal else "SECO", 17, color, True, "mt")
    for y, title, description, color, fill in (
        (475, "Passo comum", "Na casa 3, patina e permanece nela.", AMBER, AMBER_BG),
        (585, "Tração", "Na casa 3, avança uma casa.", TEAL, TEAL_BG),
    ):
        rect(d, 74, y, 1366, y + 93, fill)
        label(d, 102, y + 15, title, 33, color, True)
        label(d, 102, y + 56, description, 24, MUTED)
        label(d, 1049, y + 10, "3", 57, color, True, "mt")
        label(d, 1280, y + 10, "3" if y == 475 else "4", 57, color, True, "mt")
        arrow(d, 1102, y + 44, 1210, color)
    label(d, 76, 731, "Passo comum e esquerda: −1   ·   Tração: −3   ·   Bônus na meta: +12", 30, INK, True)
    save(im, "01-pista.png")


def approaches():
    im, d = base("02", "Três perguntas, três saídas", "O mesmo simulador permite testar objetivos distintos.")
    for i, (number, title, question, inp, out, color, fill) in enumerate([
        ("01", "Previsão", ("O que acontece se", "eu fizer isso?"), ("Posição e ação",), "Próxima posição", BLUE, BLUE_BG),
        ("02", "Imitação", ("O que o especialista", "faria aqui?"), ("Posição",), "Ação demonstrada", AMBER, AMBER_BG),
        ("03", "Planejamento", ("O que maximiza", "meu retorno?"), ("Transição + recompensa", "+ horizonte"), "Ação escolhida", TEAL, TEAL_BG),
    ]):
        x = 74 + i * 440
        rect(d, x, 294, x + 412, 758, WHITE, LINE)
        rect(d, x + 24, 318, x + 91, 368, fill, radius=15)
        label(d, x + 57, 329, number, 25, color, True, "mt")
        label(d, x + 24, 387, title, 43, color, True)
        for row, line in enumerate(question):
            label(d, x + 24, 451 + row * 35, line, 32, INK, True)
        d.line((x + 24, 538, x + 390, 538), fill=LINE, width=2)
        label(d, x + 24, 559, "USA", 21, color, True)
        for row, line in enumerate(inp):
            label(d, x + 24, 590 + row * 30, line, 27, MUTED)
        rect(d, x + 22, 680, x + 390, 728, fill, radius=14)
        label(d, x + 42, 690, out, 29, color, True)
    save(im, "02-tres-perguntas.png")


def results():
    data = json.loads((ROOT / "results.json").read_text())["results"]
    im, d = base("03", "Uma média pode esconder o erro", "Decisão em malha fechada: chegadas à meta em 32 episódios.")
    rect(d, 74, 288, 1366, 614, WHITE, LINE)
    label(d, 102, 315, "POLÍTICA", 25, MUTED, True)
    label(d, 1000, 315, "SECO", 26, BLUE, True, "mt")
    label(d, 1240, 315, "MOLHADO", 26, AMBER, True, "mt")
    for i, (key, title) in enumerate([
        ("clonagem", "Clonagem comportamental"),
        ("planejador_modelo_aprendido", "Planejamento · modelo aprendido"),
        ("planejador_modelo_exato", "Planejamento · regra exata (oráculo)"),
    ]):
        y = 355 + i * 80
        d.line((100, y, 1340, y), fill=LINE, width=2)
        label(d, 102, y + 27, title, 31, INK, i == 2)
        for x, scenario in ((1000, "seco"), (1240, "molhado")):
            n = data[scenario][key]["successes"]
            c, fill = (TEAL, TEAL_BG) if n else (AMBER, AMBER_BG)
            rect(d, x - 90, y + 13, x + 90, y + 65, fill, radius=13)
            label(d, x, y + 24, f"{n} / 32", 32, c, True, "mt")
    rect(d, 74, 648, 1366, 769, TEAL_BG)
    label(d, 102, 668, "PREDIÇÃO NO MOLHADO", 22, TEAL, True)
    pct = f'{100 * data["molhado"]["preditor"]["accuracy"]:.1f}'.replace(".", ",")
    q = data["molhado"]["preditor_pares_criticos"]
    label(d, 102, 705, f"{pct}% no agregado", 36, INK, True)
    label(d, 660, 705, f"{q['correct']} / {q['n']} acertos nos casos críticos", 36, AMBER, True)
    save(im, "03-resultados.png")


if __name__ == "__main__":
    track()
    approaches()
    results()
