"""Tools to optimize tf.keras.Model with tf-lite"""

import pathlib
import numpy as np
import tensorflow as tf

from typing import Callable, Tuple, Union, List


def get_path(path: str) -> pathlib.Path:
    save_path = pathlib.Path(path)
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path


def set_fp16_converter_flags(converter: tf.lite.TFLiteConverter):
    """Applies default optimization flags, op compatibility, and float166 quantization"""
    # https://www.tensorflow.org/lite/performance/post_training_float16_quant?hl=en
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # https://www.tensorflow.org/lite/guide/authoring?hl=it#specifying_select_tf_ops_usage
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]


def get_inference_fn(tflite_model: bytes) -> Tuple[tf.lite.Interpreter, Callable]:
    """Initializes tf-lite interpreter, and returns a function for inference.
       NOTE: works for 1-input only.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    def inference_fn(x):
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)

    return interpreter, inference_fn


class ModelOptimizer:
    def __init__(self, path: str):
        self.save_path = get_path(path)
        self.tflite_model: bytes = None

        self.converter: tf.lite.TFLiteConverter = None
        self.interpreter: tf.lite.Interpreter = None

        self.input_indices: list = None
        self.output_indices: list = None

    # def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]):
    #     """performs inference on an input batch - slow (not for deployment)"""
    #     results = [[] for _ in range(len(self.output_indices))]
    #
    #     if isinstance(inputs, (list, tuple)):
    #         assert len(inputs) == len(self.input_indices)
    #     else:
    #         inputs = [inputs]
    #
    #     for tensors in zip(*inputs):
    #         outputs = self.inference(tensors)
    #
    #         if isinstance(outputs, (list, tuple)):
    #             for i, out in enumerate(outputs):
    #                 results[i].append(out)
    #         else:
    #             results[0].append(outputs)
    #
    #     results = [np.concatenate(v) for v in results]
    #
    #     if len(results) == 1:
    #         return results[0]
    #
    #     return results

    def from_keras_model(self, model: tf.keras.Model):
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        set_fp16_converter_flags(self.converter)

    def convert(self):
        self.tflite_model = self.converter.convert()

    def save(self, file: str):
        path = self.save_path / file
        path.write_bytes(self.tflite_model)

    def interpret(self, from_path: str = None):
        if isinstance(from_path, str):
            self.interpreter = tf.lite.Interpreter(model_path=str(self.save_path / from_path))
        else:
            self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)

        self.interpreter.allocate_tensors()

        # self.input_index = self.interpreter.get_input_details()[0]['index']
        # self.output_index = self.interpreter.get_output_details()[0]['index']

        self.input_indices = [detail['index'] for detail in self.interpreter.get_input_details()]
        self.output_indices = [detail['index'] for detail in self.interpreter.get_output_details()]

    def inference(self, x: Union[tf.Tensor, np.ndarray, List[Union[tf.Tensor, np.ndarray]]]):
        if isinstance(x, (list, tuple)):
            assert len(x) == len(self.input_indices)
        else:
            x = [x]
            assert len(self.input_indices) == 1

        for input_index, tensor in zip(self.input_indices, x):
            self.interpreter.set_tensor(input_index, tensor)

        # self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        # return self.interpreter.get_tensor(self.output_index)

        outputs = [self.interpreter.get_tensor(index) for index in self.output_indices]
        assert len(outputs) == len(self.output_indices)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def resize_inputs(self, batch_size: int):
        # https://stackoverflow.com/a/53125376/21113996
        assert batch_size >= 1
        batch_size = int(batch_size)

        for index, detail in zip(self.input_indices, self.interpreter.get_input_details()):
            # substitute batch size
            shape = detail['shape']
            shape[0] = batch_size

            # set new batch size
            self.interpreter.resize_tensor_input(index, tensor_size=shape)

        self.interpreter.allocate_tensors()
