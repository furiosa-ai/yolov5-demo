import argparse
import sys
import torch
from pathlib import Path
import yaml


root_path = Path(__file__).parent


def load_model(path):
    yolo_path = root_path / "export" / "yolov5"
    sys.path.insert(0, str(yolo_path))
    model = torch.load(path, map_location="cpu")["model"].float()
    model.eval()
    del sys.path[0]
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to Pytorch weights")
    parser.add_argument("--name", required=True, help="Name to store model")
    parser.add_argument("--classes", nargs="+", help="Class names")
    args = parser.parse_args()

    input_size = (512, 512)  # (args.size, args.size)
    batch_size = 1
    out_path = root_path / "weights" / args.name

    model_file = out_path / "weights.onnx"
    cfg_file = out_path / "cfg.yaml"

    model = load_model(args.weights)

    det_layer = model.model[-1]
    det_layer.export = True

    class_names = args.classes

    if class_names is None:
        class_names = [f"cls{i}" for i in range(det_layer.nc)]

    assert len(class_names) == det_layer.nc, f"Specified number of classes does not match model. Expected: {det_layer.nc}, Got: {len(class_names)}"

    info = dict(
        anchors=det_layer.anchors.numpy().tolist(),
        class_names=class_names
    )

    x = torch.zeros(batch_size, 3, input_size[1], input_size[0])
    _ = model(x)

    out_path.mkdir(parents=True, exist_ok=False)

    torch.onnx.export(
        model, 
        x,
        model_file,
        opset_version=12,
        input_names=["input"],
        dynamic_axes={"input": {0: "b", 2: "h", 3: "w"}}
    )

    with open(cfg_file, "w") as f:
        yaml.dump(info, f)

    print(f"Exported model to {out_path.resolve()}")


if __name__ == "__main__":
    main()
