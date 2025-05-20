#!/usr/bin/env python3
"""
swap.py — Simple 128px face-swap using INSwapperONNX + InsightFace
Usage:
    python swap.py --src <your.jpg> --dst <target.jpg> --out <out.png> --device cpu|gpu
"""
import argparse
import os
import sys
import cv2


def swap_faces(
    src_path: str,
    dst_path: str,
    out_path: str,
    device: str,
):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo.inswapper import INSwapper

    providers = ["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if device == "gpu" else -1, det_size=(640, 640))

    model_path = os.path.expanduser(
        "~/.insightface/models/inswapper_128.onnx"
    )
    if not os.path.isfile(model_path):
        sys.exit(f"❌ Model file not found: {model_path}")

    session = ort.InferenceSession(model_path, providers=providers)
    swapper = INSwapper(model_file=model_path, session=session)

    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)
    if src_img is None or dst_img is None:
        sys.exit("❌ Failed to load one of the images.")

    faces_src = app.get(src_img)
    faces_dst = app.get(dst_img)
    if not faces_src or not faces_dst:
        sys.exit("❌ Could not detect a face in one of the images.")

    result = swapper.get(
        dst_img,
        faces_dst[0],
        faces_src[0],
        paste_back=True,
    )

    ext = os.path.splitext(out_path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 100] if ext in (".jpg", ".jpeg") else []

    cv2.imwrite(out_path, result, params)
    print(f"✅ Swap complete → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Face-swap 128px CLI")

    parser.add_argument(
        "--src",
        required=True,
        help="Path to your source image",
    )

    parser.add_argument(
        "--dst",
        required=True,
        help="Path to target image",
    )

    parser.add_argument(
        "--out",
        default="result.png",
        help="Output file path",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Run on CPU or GPU (needs onnxruntime-gpu)",
    )

    args = parser.parse_args()
    swap_faces(args.src, args.dst, args.out, args.device)


if __name__ == "__main__":
    main()
