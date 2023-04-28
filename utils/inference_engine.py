
import numpy as np
import onnx
from pathlib import Path

from tqdm import tqdm


def _get_onnx_input_names(model):
    input_all = [input.name for input in model.graph.input]
    input_initializer = [input.name for input in model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    return input_names


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
        if not str(onnx_file).endswith("_i8.dfg"):
            dfg_file = onnx_file.parent / (onnx_file.stem + "_i8.dfg")
        else:
            onnx_file, dfg_file = None, onnx_file

        if not dfg_file.exists():
            self.quantize(onnx_file, dfg_file, predictor.get_calib_dataset(), input_prec)

        if input_format == "hwc":
            compiler_config = {
                "permute_input": [
                    [0, 2, 3, 1],
                ],
            }
        else:
            compiler_config = None

        self.sess = session.create(str(dfg_file), device=device, compiler_config=compiler_config)

    def quantize(self, onnx_file, dfg_file, calib_dataset, input_prec, input_min=0, input_max=1):
        from furiosa.optimizer import optimize_model
        from furiosa.quantizer import quantize, Calibrator, CalibrationMethod

        model = onnx.load_model(onnx_file)
        model = optimize_model(model)
        print("optimized model")

        input_names = _get_onnx_input_names(model)

        model = model.SerializeToString()
        calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_SYM)

        for input_batch in tqdm(calib_dataset, "Computing ranges"):
            calibrator.collect_data([input_batch])

        ranges = calibrator.compute_range()
        print("Quantizing...")

        if input_prec == "i8":
            # override min and max of input tensors
            for input_name in input_names:
                ranges[input_name] = (input_min, input_max)
            graph = quantize(model, ranges, with_quantize=False)
        else:
            graph = quantize(model, ranges, with_quantize=True)
        print("Quantized model")
        graph = bytes(graph)

        with open(dfg_file, "wb") as f:
            f.write(graph)

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
