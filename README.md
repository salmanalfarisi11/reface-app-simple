---
sdk_version: gradio
app_file: app.py
---

# Simple Face-Swap App

[![CI](https://github.com/salmanalfarisi11/reface-app-simple/actions/workflows/ci.yml/badge.svg)](https://github.com/salmanalfarisi11/reface-app-simple/actions)
[![Live Demo](https://img.shields.io/badge/demo-Hugging%20Face-blue)](https://huggingface.co/spaces/salman555/reface-app-simple)
> A high-performance, 128 px face-swap prototype built with ONNX & GPU acceleration, designed for AI practitioners and developers.

---

## 📖 Overview

This repository provides a streamlined pipeline for swapping faces in still images using a 128×128 px ONNX model and the InsightFace framework. The design prioritizes:

- **Accuracy**: Leveraging InsightFace’s state-of-the-art face detector and embedding model.  
- **Performance**: Native ONNXRuntime GPU support for real-time inference.  
- **Usability**: Simple CLI and web UI via Gradio for drag-and-drop interaction.  

---

## 🚀 Features

- **CLI Interface**: `swap.py` for scripted batch processing.  
- **Web Demo**: `gradio_app.py` enables drag-and-drop face-swap in your browser.  
- **Cross-Platform**: Pure Python 3.13 compatibility on Linux, macOS, and Windows.  
- **Container-Ready**: Minimal dependencies for Docker or cloud deployment.  

---

## 🛠️ Tech Stack

- **Python 3.13**  
- **InsightFace** (buffalo_l detector + INSwapper ONNX)  
- **ONNXRuntime-GPU** for hardware acceleration  
- **OpenCV** for image I/O and post-processing  
- **Gradio** for interactive web UI  

---

## 📥 Installation

1. **Clone & enter** the project directory  
   ```bash
   git clone https://github.com/salmanalfarisi11/reface-app-simple.git
   cd reface-app-simple
   ```

2. **Create & activate** a virtual environment
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download ONNX model**
   ```bash
   mkdir -p ~/.insightface/models
   wget -O ~/.insightface/models/inswapper_128.onnx \ https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx
   ```

## ⚙️ Usage

### Command-Line
```bash
python swap_128.py \
  --src path/to/your_photo.jpg \
  --dst path/to/target.jpg \
  --out result.png \
  --device gpu
```

### WEB-UI
```bash
python gradio_app.py
```
- Open your browser at http://localhost:7860, drag-and-drop two images, and view the swapped result in real time.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/YourFeature).
3. Commit your changes (git commit -m "feat: your feature").
4. Open a Pull Request and reference any related issues.
We follow the Contributor Covenant code of conduct.
---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
