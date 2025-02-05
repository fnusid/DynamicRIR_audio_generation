from src.models.ConvTasnetTF.model import ConvTasNet
from src.models.ConvTasnetTF.param import ConvTasNetParam
import tensorflow as tf


param = ConvTasNetParam(causal=True, That=32, N=64, B=32, Sc=32, H=64)

model = ConvTasNet.make(param)
input_arr = tf.random.uniform((1, param.That, param.L))
outputs = model(input_arr)
print('Output shape', outputs.shape)
tf.saved_model.save(model, "pretrained/tflite")


def representative_dataset():
  for data in range(100):
    data = tf.random.uniform((1, param.That, param.L))
    yield [data]


converter = tf.lite.TFLiteConverter.from_saved_model("pretrained/tflite") # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

with open('pretrained/tflite/model.tflite', 'wb') as f:
  f.write(tflite_model)

# Generate tflite micro .cc
import os
os.system("xxd -i pretrained/tflite/model.tflite > model_data.cc")