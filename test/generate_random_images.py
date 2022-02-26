import numpy as np
import tensorflow as tf
from PIL import Image
import os 

images = 'noise-images'
models_path = 'models'
ESRGAN='ESRGAN.tflite'
i=0

ESRGAN_interpreter = tf.lite.Interpreter(model_path=ESRGAN)

for model in os.listdir(models_path):

	interpreter = tf.lite.Interpreter(model_path=f'{models_path}/{model}')

	noise = np.random.normal(0, 1, (1,100))
	
	noise = tf.convert_to_tensor(noise, dtype=tf.float32)
	print(f'Generate image {i} with GAN')
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]['index'], noise)
	interpreter.invoke()

	generated_image = interpreter.get_tensor(output_details[0]['index'])
	generated_image = 0.5 * generated_image + 0.5
	generated_image = generated_image * 255

	generated_image = generated_image.reshape(128,128,3).astype(np.uint8)
	Image.fromarray(generated_image).save(f'/tmp/{i}.png')
	generated_image = tf.io.read_file(f'/tmp/{i}.png')
	generated_image = tf.image.decode_jpeg(generated_image)
	generated_image = tf.expand_dims(generated_image, axis=0)
	generated_image = tf.cast(generated_image, tf.float32)

	# Get input and output tensors
	ESRGAN_interpreter.allocate_tensors()
	input_details = ESRGAN_interpreter.get_input_details()
	output_details = ESRGAN_interpreter.get_output_details()

	# Run the model
	ESRGAN_interpreter.set_tensor(input_details[0]['index'], generated_image)
	ESRGAN_interpreter.invoke()

	# Extract the output and postprocess it
	generated_image = ESRGAN_interpreter.get_tensor(output_details[0]['index'])
	generated_image = tf.squeeze(generated_image, axis=0)
	generated_image = tf.clip_by_value(generated_image, 0, 255)
	generated_image = tf.round(generated_image)
	generated_image = tf.cast(generated_image, tf.uint8)
	generated_image = generated_image.numpy()
	img = Image.fromarray(generated_image)
	img.save(f'{images}/{i}.png')
	os.remove(f'/tmp/{i}.png')
	i=i+1


