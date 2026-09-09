# ImageClassifier

ImageClassifier is a local-first toolkit for training and serving image classification models. It provides a Flask web app, command-line workflows, project-based datasets, and reusable PyTorch model utilities.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in a browser. See [GETTING_STARTED.md](GETTING_STARTED.md) for the complete workflow and [INSTALL.md](INSTALL.md) for installation details.

## Project Layout

```text
projects/<project-name>/
├── config.json
├── dataset/<class-name>/*.jpg
└── models/
    ├── model.pth
    └── class_labels.json
```

Class directory names become the model labels. Keep datasets balanced and representative.

## Roadmap

- [x] Project-based datasets and model artifacts
- [x] Browser-based upload, training, and prediction flows
- [x] CLI support for repeatable workflows
- [ ] Export trained models to ONNX with a documented dynamic-batch input contract
- [ ] Add ONNX Runtime inference for CPU-friendly deployment
- [ ] Add a lightweight real-time camera inference endpoint and example client
- [ ] Bundle preprocessing metadata and class labels with exports
- [ ] Add automated tests for export and inference parity

## License

ImageClassifier is distributed under the MIT License. See [LICENSE](LICENSE).
