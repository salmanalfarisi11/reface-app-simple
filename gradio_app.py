#!/usr/bin/env python3
"""
gradio_app.py — Drag-and-drop 128px face-swap web UI with Gradio
Usage:
    python gradio_app.py
"""
import os
import numpy as np
import cv2
import gradio as gr


def get_models():
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo.inswapper import INSwapper

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    model_path = os.path.expanduser(
        "~/.insightface/models/inswapper_128.onnx"
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(model_path, providers=providers)
    swapper = INSwapper(model_file=model_path, session=session)

    return app, swapper


def swap_fn(src, dst):
    app, swapper = get_models()

    src_img = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    dst_img = cv2.cvtColor(np.array(dst), cv2.COLOR_RGB2BGR)

    faces_src = app.get(src_img)
    faces_dst = app.get(dst_img)
    if not faces_src or not faces_dst:
        return None, "❌ Face not detected in one of the images."

    result = swapper.get(dst_img, faces_dst[0], faces_src[0], paste_back=True)
    out_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return out_img, "✅ Success"


iface = gr.Interface(
    fn=swap_fn,
    inputs=[
        gr.Image(type="pil", label="Source (your face)"),
        gr.Image(type="pil", label="Target (celebrity)"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Swapped Result"),
        gr.Textbox(label="Status"),
    ],
    title="Simple 128px Face-Swap",
    description="Upload two images and get a face-swapped result.",
    allow_flagging="never",
    live=False,
)


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
