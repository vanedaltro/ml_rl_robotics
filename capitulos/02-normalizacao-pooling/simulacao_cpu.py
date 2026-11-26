"""MuJoCo: meça física, câmera e inferência separadamente.
Linux headless (Mesa): MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 python simulacao_cpu.py
"""
import argparse
import csv
import random
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from OpenGL.GL import GL_RENDERER, glGetString
from benchmark_cpu import modelo, fundir

SCENE = """
<mujoco model="bench bancada">
  <option timestep="0.005"/>
  <visual><global offwidth="64" offheight="64"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="bancada" pos="0 0 2.3"/>
    <geom type="plane" size="2 2 .1" rgba=".27 .32 .39 1"/>
    <body pos=".22 -.10 .35"><freejoint/>
      <geom type="box" size=".12 .12 .12" rgba=".82 .45 .18 1"/>
    </body>
    <body pos="-.25 .18 .15">
      <geom type="cylinder" size=".13 .15" rgba=".33 .67 .92 1"/>
    </body>
  </worldbody>
</mujoco>"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--saida", type=Path, default=Path("simulacao_cpu.csv"))
    args = p.parse_args()
    if args.frames < 1 or args.reps < 1:
        p.error("Use frames e repetições positivos.")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    scene = mujoco.MjModel.from_xml_string(SCENE)
    state = mujoco.MjData(scene)
    bn = modelo("batch+max").eval()
    nets = {"batch+max": bn, "batch_fundida": fundir(bn),
            "layer_c+max": modelo("layer_c+max"), "batch+stride": modelo("batch+stride")}
    for net in nets.values():
        net.eval()
    coords = np.linspace(0, 63, 28, dtype=int)
    weights = np.array([.299, .587, .114], dtype=np.float32)
    rows = []
    with mujoco.Renderer(scene, height=64, width=64) as renderer, torch.inference_mode():
        mujoco.mj_forward(scene, state)
        renderer.update_scene(state, camera="bancada")
        a = renderer.render()
        assert a.shape == (64,64,3) and a.max() > a.min()
        dev = glGetString(GL_RENDERER)
        print(f"MuJoCo {mujoco.__version__} | renderizador {dev.decode() if dev else 'não identificado'}", flush=True)
        idxs = np.ix_(coords, coords)
        names = list(nets)
        now = time.perf_counter_ns
        for rep in range(args.reps):
            mujoco.mj_resetData(scene, state)
            mujoco.mj_forward(scene, state)
            ordem_rng = random.Random(rep + 891)
            timings = {k: [] for k in names}
            for i in range(args.frames + 15):
                t0=now()
                mujoco.mj_step(scene, state)
                t1=now()
                renderer.update_scene(state, camera="bancada")
                pixels=renderer.render()
                t2=now()
                gray = (pixels[idxs] @ weights) / 255.
                x = torch.from_numpy(gray)[None, None]
                t3=now()
                common = ((t1-t0)/1e6, (t2-t1)/1e6, (t3-t2)/1e6)
                ordem = names.copy()
                ordem_rng.shuffle(ordem)
                for name in ordem:
                    start = now()
                    nets[name](x)
                    step = (now()-start)/1e6
                    if i >= 15:
                        timings[name].append([*common,step])
            for name in names:
                runs = np.array(timings[name])
                total = runs.sum(axis=1)
                values = [*np.median(runs, axis=0), np.median(total), np.quantile(total,.95)]
                rows.append([rep,name,*values])
                print(f"{rep} {name:14}", *[round(x,5) for x in values], flush=True)
    with args.saida.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep","modelo","fisica_ms","render_ms","preproc_ms","modelo_ms","total_ms","total_p95_ms"])
        w.writerows(rows)
    print("Mediana das repetições: física, câmera, preparo, inferência, total, p95 total (ms)")
    for name in nets:
        print(name, *[round(x,5) for x in np.median([r[2:] for r in rows if r[1]==name],axis=0)])

if __name__ == "__main__":
    main()
