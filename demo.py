
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from utils.video_record import VideoRecord
from models.yolov5 import Yolov5Detector

import argparse
import random
import cv2
from tqdm import tqdm
import numpy as np

from utils.image_input import ImageInput


COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    tl = line_thickness or round(
        0.0004 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        score = box[4]
        class_id = int(box[5])

        # box text and bar
        color = COLORS_10[class_id % len(COLORS_10)]
        label = f"{identities[class_id]} ({score:.2f})"
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label)
    return img


def file_path(path_type="file_dir", ext=None, force_exist=True):
    assert path_type in ("file", "dir", "file_dir")

    if ext is not None:
        path_type = "file"

    def _checker(path):
        path = Path(path)

        if path_type == "file":
            if not path.is_file():
                raise FileNotFoundError(path)
        elif path_type == "dir":
            if not path.is_dir():
                raise NotADirectoryError(path)
        elif path_type == "file_dir":
            if not path.exists():
                raise FileNotFoundError(path)

        if ext is not None:
            if path.suffix != ext:
                raise Exception(f"{path} is not a {ext} file")

        return path

    return _checker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=file_path(), help="Path to image / image folder / video file")
    parser.add_argument("--model", required=True, type=file_path(path_type="dir"), help="Path to onnx weights folder")
    parser.add_argument("--framework", required=True, choices=("onnx", "furiosa"), help="Which backend to use")
    parser.add_argument("--calib-data", default="../data/bdd100k/images/100k/test", type=file_path(path_type="dir"), help="Path to calibration data containing image files")
    parser.add_argument("--calib-data-count", default=10, type=int, help="How many images to use for calibration")
    parser.add_argument("--no-display", action="store_true", help="Disables displaying results on the screen (useful for server)")
    parser.add_argument("--frame-limit", type=int, help="Stops inference after frame limit is reached")
    parser.add_argument("--record-file", help="Record results to specified video file (e.g. \"out.mp4\")")
    args = parser.parse_args()

    frame_limit = args.frame_limit
    record_file = Path(args.record_file) if args.record_file is not None else None
    input_reader = ImageInput.create(args.input)
    output_writer = VideoRecord(str(record_file)) if record_file is not None else None

    model_path = Path(args.model)
    framework = args.framework
    weights = model_path / "weights.onnx"
    cfg_file = model_path / "cfg.yaml"
    calib_data = args.calib_data
    calib_data_count = args.calib_data_count

    detector = Yolov5Detector(weights, cfg_file, framework, calib_data, calib_data_count)

    frame_count = frame_limit if frame_limit is not None else len(input_reader)
    input_is_img = len(input_reader) == 1

    for i, img in tqdm(enumerate(input_reader), total=frame_count):
        if frame_limit is not None and i >= frame_limit:
            break

        boxes = detector(img)

        img = draw_bboxes(img, boxes, detector.class_names)

        if output_writer is not None:
            output_writer.update(img)

        if input_is_img:
            cv2.imwrite("result.jpg", img)

        if not args.no_display:
            cv2.imshow("detect", img)
            cv2.waitKey(1 if not input_is_img else 0)
    
    if output_writer is not None:
        output_writer.close()
    detector.close()

    input_reader.close()


if __name__ == "__main__":
    main()
