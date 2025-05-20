#!/usr/bin/env python3
"""
swap_128.py  —  Face‐swap pakai InswapperONNX 128 px + InsightFace (Python 3.13 ready)
Usage:
    python swap_128.py \
      --src me.jpg \
      --dst target.jpg \
      --out result.png \
      --device cpu|gpu
"""
import argparse, os, sys, cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

def load_models(device: str):
    # 1) Pilih providers untuk onnxruntime
    providers = (["CUDAExecutionProvider"] if device=="gpu"
                 else ["CPUExecutionProvider"])

    # 2) FaceAnalysis: deteksi + landmark + embedding (buffalo_l)
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if device=="gpu" else -1, det_size=(640,640))

    # 3) InSwapperONNX (model 128px)
    model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
    if not os.path.isfile(model_path):
        sys.exit(f"❌ File model tidak ditemukan:\n   {model_path}\n"
                 "   Silakan unduh inswapper_128.onnx ke folder ini.")
    session = ort.InferenceSession(model_path, providers=providers)
    swapper = INSwapper(model_file=model_path, session=session)

    return app, swapper

def swap_faces(src_path, dst_path, out_path, device):
    # validasi file ada
    for p in (src_path, dst_path):
        if not os.path.isfile(p):
            sys.exit(f"❌ File tidak ditemukan: {p}")

    app, swapper = load_models(device)

    # baca gambar
    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)

    # deteksi wajah
    faces_src = app.get(src_img)
    faces_dst = app.get(dst_img)
    if not faces_src or not faces_dst:
        sys.exit("❌ Gagal mendeteksi wajah di salah satu gambar.")

    # swap + langsung paste-back
    result = swapper.get(dst_img, faces_dst[0], faces_src[0], paste_back=True)

    # simpan: lossless PNG atau JPEG q=100
    ext = os.path.splitext(out_path)[1].lower()
    params = ([cv2.IMWRITE_JPEG_QUALITY, 100]
              if ext in (".jpg",".jpeg") else [])
    cv2.imwrite(out_path, result, params)
    print(f"✅ Face‐swap selesai, tersimpan di: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Simple Face‐swap 128px")
    p.add_argument("--src",    required=True, help="Path ke foto sumber (anda)")
    p.add_argument("--dst",    required=True, help="Path ke foto target")
    p.add_argument("--out",    default="result.png", help="Path keluaran")
    p.add_argument("--device", choices=["cpu","gpu"], default="cpu",
                   help="Pakai CPU atau GPU (butuh onnxruntime-gpu)")
    args = p.parse_args()
    swap_faces(args.src, args.dst, args.out, args.device)

if __name__=="__main__":
    main()
