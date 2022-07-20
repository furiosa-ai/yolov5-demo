import numpy as np
import cv2
import yaml
from utils.box_decode.box_decoder import BoxDecoderPytorch, BoxDecoderC
from utils.nms import nms

from utils.inference_engine import InferenceEngineOnnx
from utils.transforms import letterbox


class Yolov5Detector:
    def __init__(self, model_file, cfg_file, framework, conf_thres=0.25, iou_thres=0.45, 
        input_color_format="bgr", box_decoder="c") -> None:
        if framework == "onnx":
            input_format = "chw"
            input_type = "f32"
            infer = InferenceEngineOnnx(model_file)

        assert input_format in ("chw", "hwc")
        assert input_type in ("f32", "i8")
        assert input_color_format in ("rgb", "bgr")
        assert box_decoder in ("pytorch", "c")

        if input_format == "chw":
            b, c, h, w = infer.get_input_shapes()[0]
        elif input_format == "hwc":
            b, h, w, c = infer.get_input_shapes()[0]

        self.infer = infer
        self.input_size = w, h

        self.input_format = input_format
        self.input_type = input_type
        self.input_color_format = input_color_format

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

    def preproc(self, img):
        img, (sx, sy), (padw, padh) = self._resize(img)

        if self.input_color_format == "bgr":
            img = self._cvt_color(img)

        if self.input_format == "chw":
            img = self._transpose(img)

        if self.input_type == "f32":
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

    def close(self):
        self.infer.close()
