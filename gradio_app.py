#!/usr/bin/env python3
# gradio_app.py ── Drag&drop face-swap 128px via web UI

import gradio as gr
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
import onnxruntime as ort
import os

# Load models once
providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))
model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
session = ort.InferenceSession(model_path, providers=providers)
swapper = INSwapper(model_file=model_path, session=session)

def swap_fn(src, dst):
    # src/dst = PIL images → convert ke OpenCV BGR
    src_img = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    dst_img = cv2.cvtColor(np.array(dst), cv2.COLOR_RGB2BGR)

    fs, fd = app.get(src_img), app.get(dst_img)
    if not fs or not fd:
        return None, "Face not detected in one of the images."

    res = swapper.get(dst_img, fd[0], fs[0], paste_back=True)
    # convert kembali ke RGB PIL
    out = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return out, "✅ Success"

# Gradio Interface
title = "🖼 Simple Face-Swap 128px"
descr = "Upload **Source** + **Target**, pilih CPU/GPU, lalu klik **RUN**"
iface = gr.Interface(
    fn=swap_fn,
    inputs=[
        gr.Image(type="pil", label="Source (you)"),
        gr.Image(type="pil", label="Target")
    ],
    outputs=[
        gr.Image(type="numpy", label="Swapped Result"),
        gr.Textbox(label="Status")
    ],
    title=title,
    description=descr,
    allow_flagging="never",
    live=False
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
