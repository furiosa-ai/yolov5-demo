import tempfile
import numpy as np
import yaml
import os


class InferenceEngine:
    def __init__(self) -> None:
        pass

    def get_input_shapes(self):
        raise NotImplementedError

    def get_output_shapes(self):
        raise NotImplementedError

    def infer(self, *x):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __call__(self, *x):
        return self.infer(*x)


class InferenceEngineOnnx(InferenceEngine):
    def __init__(self, onnx_file, input_names=None, output_names=None) -> None:
        super().__init__()
        import onnxruntime

        self.ort_session = onnxruntime.InferenceSession(str(onnx_file))

        if input_names is None:
            input_names = [i.name for i in self.ort_session.get_inputs()]

        if output_names is None:
            output_names = [i.name for i in self.ort_session.get_outputs()]

        self.input_names = input_names
        self.output_names = output_names

    def get_input_shapes(self):
        inputs = self.ort_session.get_inputs()
        shapes = [i.shape for i in inputs]
        return shapes

    def get_output_shapes(self):
        outputs = self.ort_session.get_outputs()
        shapes = [i.shape for i in outputs]
        return shapes

    def infer(self, *x):
        if len(x) == 1 and isinstance(x, dict):
            input_dict = x[0]
        else:
            input_dict = {k: v for k, v in zip(self.input_names, x)}
        out = self.ort_session.run(self.output_names, input_dict)

        return out

    def close(self):
        pass

