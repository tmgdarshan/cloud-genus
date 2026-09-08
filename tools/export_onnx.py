# -*- coding: utf-8 -*-
"""
Export the CCSN 11-class classifier to ONNX for on-device (browser) inference.

Each of the 3 seed checkpoints becomes its own graph that outputs raw logits.
The app runs all 3, softmaxes, and averages (the ensemble).

We emit float16 (keep_io_types=True, so JS still feeds/reads float32). float16 halves
the size and is fully supported by onnxruntime-web's WASM backend — unlike int8 dynamic
quantization, whose ConvInteger op is not supported in ort-web and silently fails to load.

Input : float32 [1, 3, 224, 224]  — resized + ImageNet-normalized (done in JS)
Output: float32 [1, 11]            — logits

    python app/tools/export_onnx.py
Outputs: app/model/seed{42,43,44}.fp16.onnx, app/model/labels.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxconverter_common import float16
from torchvision import models

CKPT_DIR = Path("D:/cloud-genus-app/models")
OUT = Path(__file__).resolve().parents[1] / "model"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]

CLASSES = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
NAMES = ["Altocumulus", "Altostratus", "Cumulonimbus", "Cirrocumulus", "Cirrus",
         "Cirrostratus", "Contrail", "Cumulus", "Nimbostratus", "Stratocumulus", "Stratus"]


def build(num_classes: int = 11) -> nn.Module:
    m = models.resnet18(weights=None)
    f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(f, 256), nn.BatchNorm1d(256),
        nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_classes),
    )
    return m


def load(seed: int) -> nn.Module:
    net = build()
    ck = torch.load(CKPT_DIR / f"resnet18_ccsn11_seed{seed}.pth", map_location="cpu")
    net.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    return net.eval()


def main() -> None:
    dummy = torch.randn(1, 3, 224, 224)
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((8, 3, 224, 224)).astype(np.float32)
    ens_ref = np.zeros((8, 11)); ens_ort = np.zeros((8, 11))

    for s in SEEDS:
        net = load(s)
        with torch.no_grad():
            ref = net(torch.from_numpy(xs)).numpy()
        ens_ref += torch.softmax(torch.from_numpy(ref), 1).numpy()

        tmp = OUT / f"seed{s}.fp32.onnx"
        torch.onnx.export(
            net, dummy, str(tmp),
            input_names=["input"], output_names=["logits"],
            opset_version=13, dynamo=False,
        )
        m16 = float16.convert_float_to_float16(onnx.load(str(tmp)), keep_io_types=True)
        dst = OUT / f"seed{s}.fp16.onnx"
        onnx.save(m16, str(dst))
        tmp.unlink()

        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        got = np.concatenate([sess.run(None, {"input": xs[i:i + 1]})[0] for i in range(8)], 0)
        ens_ort += torch.softmax(torch.from_numpy(got), 1).numpy()
        print(f"[+] {dst.name}  {dst.stat().st_size / 1e6:.1f} MB  "
              f"(logit mean|Δ| {np.abs(got - ref).mean():.4f})")

    ens_ref /= len(SEEDS); ens_ort /= len(SEEDS)
    print(f"[=] ensemble: mean|Δprob| {np.abs(ens_ort - ens_ref).mean():.5f}  "
          f"argmax-agree {(ens_ort.argmax(1) == ens_ref.argmax(1)).mean():.2f}")

    for old in OUT.glob("seed*.int8.onnx"):
        old.unlink()
    (OUT / "labels.json").write_text(json.dumps(
        [{"code": c, "name": n} for c, n in zip(CLASSES, NAMES)], indent=1))
    print(f"[+] labels.json  ({', '.join(CLASSES)})")


if __name__ == "__main__":
    main()
