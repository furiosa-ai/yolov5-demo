import random
import numpy as np
import cv2
import yaml
from models.predictor import Predictor
from utils.box_decode.box_decoder import BoxDecoderPytorch, BoxDecoderC
from utils.nms import nms

from utils.inference_engine import InferenceEngineOnnx, InferenceEngineFuriosa
from utils.transforms import letterbox


class Yolov5Detector(Predictor):
    def __init__(self, model_file, cfg_file, framework, calib_data_path=None, calib_data_count=None, conf_thres=0.25, iou_thres=0.45, 
        input_color_format="bgr", box_decoder="c") -> None:

        self.input_color_format = input_color_format
        self.calib_data_path = calib_data_path
        self.calib_data_count = calib_data_count

        if framework == "furiosa":
            input_format = "hwc"
            input_prec = "i8"
        elif framework == "onnx":
            input_format = "chw"
            input_prec = "f32"

        self.input_format = input_format
        self.input_prec = input_prec

        # load input name and shape in advance from onnx file
        input_name, input_shape = Yolov5Detector._get_input_name_shape(model_file)
        b, c, h, w = input_shape
        assert b == 1, "Code only supports batch size 1"

        self.input_name = input_name
        self.input_size = w, h

        if framework == "furiosa":
            infer = InferenceEngineFuriosa(self, model_file)
        elif framework == "onnx":
            infer = InferenceEngineOnnx(self, model_file)

        assert input_format in ("chw", "hwc")
        assert input_prec in ("f32", "i8")
        assert input_color_format in ("rgb", "bgr")
        assert box_decoder in ("pytorch", "c")

        self.infer = infer

        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
            self.anchors = np.float32(cfg["anchors"])
            self.class_names = cfg["class_names"]
        
        self.stride = self._compute_stride()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if box_decoder == "pytorch":
            box_decoder = BoxDecoderPytorch(nc=len(self.class_names), anchors=self.anchors, stride=self.stride, conf_thres=self.conf_thres)
        elif box_decoder == "c":
            box_decoder = BoxDecoderC(nc=len(self.class_names), anchors=self.anchors, stride=self.stride, conf_thres=self.conf_thres)

        self.box_decoder = box_decoder

    @staticmethod
    def _get_input_name_shape(onnx_file):
        temp_sess = InferenceEngineOnnx(None, onnx_file)
        input_shape = temp_sess.get_input_shapes()[0]
        input_name = temp_sess.input_names[0]

        return input_name, input_shape

    def get_class_count(self):
        return len(self.class_names)

    def get_output_feat_count(self):
        return self.anchors.shape[0]

    def get_anchor_per_layer_count(self):
        return self.anchors.shape[1]

    def _compute_stride(self):
        img_h = self.input_size[1]
        feat_h = np.float32([shape[2] for shape in self.infer.get_output_shapes()])
        strides = img_h / feat_h
        return strides

    def _resize(self, img):
        w, h = self.input_size
        return letterbox(img, (h, w), auto=False)

    def _cvt_color(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _transpose(self, img):
        return img.transpose(2, 0, 1)

    def _normalize(self, img):
        img = img.astype(np.float32) / 255
        return img

    def _reshape_output(self, feat):
        return np.ascontiguousarray(feat.reshape(
            feat.shape[0], self.get_anchor_per_layer_count(), self.get_class_count() + 5, feat.shape[2], feat.shape[3]
        ).transpose(0, 1, 3, 4, 2))

    def preproc(self, img, input_format=None, input_prec=None):
        if input_format is None:
            input_format = self.input_format

        if input_prec is None:
            input_prec = self.input_prec

        img, (sx, sy), (padw, padh) = self._resize(img)

        if self.input_color_format == "bgr":
            img = self._cvt_color(img)

        if input_format == "chw":
            img = self._transpose(img)

        if input_prec == "f32":
            img = self._normalize(img)

        assert sx == sy
        scale = sx

        return img, (scale, (padw, padh))

    def postproc(self, feats_batched, preproc_params):
        boxes_batched = []

        for i, (scale, (padw, padh)) in enumerate(preproc_params):
            feats = [f[i:i+1] for f in feats_batched]
            feats = [self._reshape_output(f) for f in feats]
            boxes = self.box_decoder(feats)
            boxes = nms(boxes, self.iou_thres)[0]

            # rescale boxes
            boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
            boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

            boxes_batched.append(boxes)

        return boxes_batched

    def __call__(self, imgs):
        single_input = not isinstance(imgs, (tuple, list))
        if single_input:
            imgs = [imgs]

        inputs, preproc_params = zip(*[self.preproc(img) for img in imgs])
        inputs = np.stack(inputs)
        feats = self.infer(inputs)
        res = self.postproc(feats, preproc_params)

        if single_input:
            res = res[0]

        return res

    def get_calib_dataset(self):
        calib_files = sorted(self.calib_data_path.glob("*"))
        random.Random(123).shuffle(calib_files)  # shuffle with seed

        assert len(calib_files) > 0, f"no calib files found at {self.calib_data_path}"

        if self.calib_data_count is not None:
            calib_files = calib_files[:self.calib_data_count]

        def _load_data(file):
            img = cv2.imread(str(file))
            inputs, _ = self.preproc(img, input_format="chw", input_prec="f32")
            inputs = inputs[None]  # insert batch dim
            return [inputs]

        return (_load_data(f) for f in calib_files)

    def close(self):
        self.infer.close()
