import tensorflow as tf
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 128, 128, 3], dtype=tf.float32)])
def f(input):
  return concrete_func(input);

# Converting ConcreteFunctions to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_concrete_functions([f.get_concrete_function()])
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('ESRGAN.tflite', 'wb') as f:
  f.write(tflite_model)


