# Hugging Face deployment

This repository now has two deployment modes:

- **Local PyTorch mode:** run `python app.py`. This keeps the existing Flask API, filesystem projects, and PyTorch training in `app.py` and `scripts/train_model.py`.
- **Browser-local Space mode:** publish the contents of `spaces/` to a Hugging Face Docker Space. The browser reads the dataset, trains with TensorFlow.js, displays metrics, and runs inference without sending images to the Space.

## Publish the browser Space

Create a new Hugging Face Space with the **Docker** SDK, then copy the contents of `spaces/` into that Space repository:

```bash
git clone https://huggingface.co/spaces/<account>/<space-name>
cp spaces/Dockerfile spaces/README.md spaces/index.html spaces/styles.css spaces/app.js <space-name>/
cd <space-name>
git add .
git commit -m "Add local browser image trainer"
git push
```

The Space listens on port `7860`. No Python ML dependencies are required by this browser-only deployment. The TensorFlow.js and ONNX Runtime Web scripts are loaded from jsDelivr, so the browser needs network access on first load.

## ONNX export

ONNX is an inference interchange format here; it does not replace browser training. To export an existing local PyTorch project:

```bash
pip install -r requirements-onnx.txt
python scripts/export_onnx.py --project Testing_App
```

This creates `projects/Testing_App/models/model.onnx` and `class_labels.json`. In the Space, load those files to run browser-side ONNX inference. The exported model expects RGB `128x128` input normalized to `[-1, 1]`, matching the current PyTorch predictor.

## Important limitations

- WebGPU availability depends on the visitor's browser and device. TensorFlow.js falls back to another browser backend when needed.
- Training large datasets can exhaust browser memory; this demo keeps all decoded tensors in memory.
- Browser models are stored only in the current session unless the user downloads them.
- A public Space can host the interface for free, but it cannot guarantee a fixed amount of the user's CPU/GPU usage.
- The browser app is intentionally separate from the Flask app. Do not replace the local `app.py` deployment with the Space files.
