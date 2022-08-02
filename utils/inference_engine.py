
import numpy as np
import onnx
from pathlib import Path


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
    def __init__(self, predictor, onnx_file, input_names=None, output_names=None) -> None:
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


class InferenceEngineFuriosa(InferenceEngine):
    def __init__(self, predictor, onnx_file, device=None) -> None:
        super().__init__()

        from furiosa.runtime import session

        input_prec = predictor.input_prec
        input_format = predictor.input_format

        onnx_file = Path(onnx_file)
        if not str(onnx_file).endswith("_i8.onnx"):
            onnx_i8_file = onnx_file.parent / (onnx_file.stem + "_i8" + onnx_file.suffix)
        else:
            onnx_file, onnx_i8_file = None, onnx_file

        if not onnx_i8_file.exists():
            self.quantize(onnx_file, onnx_i8_file, predictor.get_calib_dataset())

        if not (input_prec == "f32" and input_format == "chw"):
            assert input_prec in ("f32", "i8")
            assert input_format in ("chw", "hwc")

            assert not (input_prec == "f32" and input_format != "chw")
            assert input_prec == "i8", "Nothing to do"

            compile_config = {
                "without_quantize": {
                    "parameters": [
                        {
                            "input_min": 0.0, "input_max": 1.0, 
                            "permute": [0, 2, 3, 1] if input_format == "hwc" else [0, 1, 2, 3]
                        }
                    ]
                },
            }
        else:
            compile_config = None

        self.sess = session.create(str(onnx_i8_file), device=device, compile_config=compile_config)

    def quantize(self, onnx_file, onnx_i8_file, calib_dataset):
        from furiosa.quantizer.frontend.onnx import optimize_model
        from furiosa.quantizer.frontend.onnx.calibrate import calibrate
        from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
        from furiosa.quantizer.frontend.onnx.quantizer import quantizer

        model = onnx.load_model(onnx_file)
        optimized_model = optimize_model(model)
        print("optimized model")

        dynamic_ranges = calibrate(optimized_model, calib_dataset)

        quant_model = quantizer.FuriosaONNXQuantizer(
            optimized_model, True, True, QuantizationMode.DFG, dynamic_ranges
        ).quantize()

        onnx.save_model(quant_model, onnx_i8_file)

    def get_input_shapes(self):
        inputs = self.sess.inputs()
        shapes = [i.shape for i in inputs]
        return shapes

    def get_output_shapes(self):
        outputs = self.sess.outputs()
        shapes = [i.shape for i in outputs]
        return shapes

    def infer(self, *x):
        x = list(x)
        outputs = self.sess.run(x)

        outputs = [outputs[i].numpy() for i in range(len(outputs))]

        return outputs

    def close(self):
        self.sess.close()
