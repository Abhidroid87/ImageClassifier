"""Export a locally trained project model for browser-side ONNX inference."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import ImageClassifier


def export_project(project_name: str, output_path: Optional[str] = None) -> Path:
    project_dir = Path("projects") / project_name
    config_path = project_dir / "config.json"
    model_path = project_dir / "models" / "model.pth"

    with config_path.open() as config_file:
        config = json.load(config_file)

    labels = config.get("classes", [])
    if not labels:
        raise ValueError("The project has no class labels.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    model = ImageClassifier(num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    destination = Path(output_path) if output_path else project_dir / "models" / "model.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model,
        sample,
        destination,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    labels_path = destination.with_name("class_labels.json")
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Exported ONNX model: {destination}")
    print(f"Exported labels: {labels_path}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Project name under projects/")
    parser.add_argument("--output", help="Optional output .onnx path")
    arguments = parser.parse_args()
    export_project(arguments.project, arguments.output)
