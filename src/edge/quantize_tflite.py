import tensorflow as tf
import numpy as np

def representative_data_gen():
  for i in range(100):
    mix = np.random((1, 2, 160 + 64))
    emb = np.random((1, 1, 128))
    enc_buf = np.random((1, 128, 2046))
    dec_buf = np.random((1, 128, 3))
    out_buf = np.random((1, 128, 513))
    
    yield [mix, emb, enc_buf, dec_buf, out_buf]

converter = tf.lite.TFLiteConverter.from_saved_model('pretrained/tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()

# save the converted model 
open('pretrained/tf/tse_int8.tflite', 'wb').write(tflite_quant_model)