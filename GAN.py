import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, BatchNormalization, Reshape, Dropout, Dense, Flatten
from tensorflow.keras.layers import Activation, ZeroPadding2D, UpSampling2D, Conv2D, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import os

# Change/Remove as necessary
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

momentum = 0.8
alpha = 0.2
dropoutRate =0.5
generated_images_path='generated_images'

# Save rows x cols samples to image, takes in epoch number for naming purposes and noise
def save_image(epoch, noise, generator, imgShape):
	rows = 5
	cols = 5
	margin = 16
	image_array = np.full(( 
	  margin + (rows * (imgShape+margin)), 
	  margin + (cols * (imgShape+margin)), 3), 
	  255, dtype=np.uint8)
	generated_images = generator.predict(noise)
	generated_images = 0.5 * generated_images + 0.5
	image_count = 0

	for row in range(rows):
		for col in range(cols):
			r = row * (imgShape+16) + margin
			c = col * (imgShape+16) + margin
			image_array[r:r+imgShape,c:c+imgShape] = generated_images[image_count] * 255
			image_count += 1

	im = Image.fromarray(image_array)
	im.save(f"{generated_images_path}/sample" + str(epoch) + ".png")

# Load training data from image set, takes in boolean loadNewData, inputPath for images, 
# imageShape, and dataSavePath for both saving and reading numpy array 
def load_train_data(loadNewData, inputPath, imageShape, dataSavePath):
	trainingData = []
	
	if loadNewData:
		print('Loading training images...')
		data_augmentation = tf.keras.Sequential([
	    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
	    layers.experimental.preprocessing.RandomRotation(0.4)])
		for filename in sorted(glob.glob(inputPath)):
			print(f'Processing {filename}')
			try:
				image = Image.open(filename).resize((imageShape[0], imageShape[1]), Image.ANTIALIAS)
				if image.mode=='RGBA':
					image = image.convert('RGB')
				if np.asarray(image).shape == imageShape:
					trainingData.append(np.asarray(image))
					image = tf.expand_dims(image, 0)
					for i in range(5):
					  augmented_image = data_augmentation(image)
					  trainingData.append(np.asarray(augmented_image[0]))
				else:
					print(f'Invalid image {filename}')
				
			except Exception as ex:
				print(ex)
				continue
		
		print('Training data size:', len(trainingData))	
		
		#reshape training data to the expected input for the model 
		trainingData = np.reshape(trainingData, (-1, imageShape[0], imageShape[1], imageShape[2]))

		#scale data 
		trainingData = trainingData / 127.5 - 1
		np.save(dataSavePath, trainingData)
	else:
		trainingData = np.load(dataSavePath)

	return trainingData


def build_generator(noiseShape):
	model = Sequential()
	model.add(Dense(8*8*256, use_bias=False, input_shape=noiseShape))
	model.add(BatchNormalization(momentum=momentum))
	model.add(LeakyReLU())

	model.add(Reshape((8, 8, 256)))
	assert model.output_shape == (None, 8, 8, 256) #is the batchsize 
	model.add(LeakyReLU())

	model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 16, 16, 256)
	model.add(BatchNormalization(momentum=momentum))
	model.add(LeakyReLU())

	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 32, 32, 128)
	model.add(BatchNormalization(momentum=momentum))
	model.add(LeakyReLU())

	model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 64, 64, 64)
	model.add(BatchNormalization(momentum=momentum))
	model.add(LeakyReLU())

	model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 128, 128, 3)

	print('Model: Generator')
	model.summary()

	noise = Input(shape = noiseShape)
	image = model(noise)

	return Model(noise, image)


def build_discriminator(imageShape):
	model = Sequential()
	model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same', input_shape=imageShape))
	model.add(LeakyReLU(alpha))
	model.add(Dropout(dropoutRate))

	model.add(Conv2D(512, (4, 4), strides=(2, 2), padding='same', input_shape=imageShape))
	model.add(LeakyReLU(alpha))
	model.add(Dropout(dropoutRate))

	model.add(Conv2D(1024, (4, 4), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha))
	model.add(Dropout(dropoutRate))

	model.add(Flatten())
	model.add(Dense(1, activation="sigmoid"))

	print('Model: Discriminator')
	model.summary()

	image = Input(shape = imageShape)
	validity = model(image)
	
	return Model(image, validity)



# Main training function. Takes in epochs, batchSize, learningRate, 
# saveFreq (number of epochs between saves), loadNewData (set to false if no numpy file exists), and numGPUs
def train(noiseShape, imageShape, epochs, batchSize, learningRate, saveFreq, inputPath, loadNewData, numGPUs):

	optimizer = Adam(learningRate, 0.5)

	print("Building Discriminator")
	discriminator = build_discriminator(imageShape)
#	discriminator = tf.keras.utils.multi_gpu_model(discriminator, gpus = numGPUs)
	discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

	print("Building Generator")
	generator = build_generator(noiseShape)
#	generator = tf.keras.utils.multi_gpu_model(generator, gpus = numGPUs)
	noise = Input(shape = noiseShape)
	image = generator(noise)

	discriminator.trainable = False

	valid = discriminator(image)

	print("Building Combined")
	combined = Model(noise, valid)
#	combined = tf.keras.utils.multi_gpu_model(combined, gpus = numGPUs)
	combined.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])

	trainingData = load_train_data(loadNewData = loadNewData, inputPath = inputPath, 
		imageShape = imageShape, dataSavePath = "training_data.npy");

	print(trainingData.shape)

	#tf.keras.backend.get_session().run(tf.global_variables_initializer())

	for epoch in range(1, epochs):
		y_real = np.ones((batchSize, 1))
		y_fake = np.zeros((batchSize, 1))
		
		idx = np.random.randint(0, trainingData.shape[1], batchSize)
		x_real = trainingData[idx]

		noise = np.random.normal(0, 1, (batchSize, noiseShape[0]))
		x_fake = generator.predict(noise)
		
		discriminator_real = discriminator.train_on_batch(x_real, y_real) # 
		discriminator_fake = discriminator.train_on_batch(x_fake, y_fake)

		discriminator_metric = .5 * np.add(discriminator_real, discriminator_fake)

		valid_y = np.array([1] * batchSize)
		noise = np.random.normal(0, 1, (batchSize, noiseShape[0]))
		generator_metric = combined.train_on_batch(noise, valid_y)

		print("Epoch:", epoch, "d_loss:", discriminator_metric[0], "g_loss:", generator_metric[0], "d_loss_fake:", discriminator_fake[0], "d_loss_real:", discriminator_real[0])

		if epoch % saveFreq == 0:
			save_image(epoch, noise, generator, imageShape[0])
			generator.save(f"models/generator_{epoch}.h5")

	generator.save(f"models/generator_{epoch}.h5")
			
train(noiseShape=(100, ), 
	imageShape=(128, 128, 3),
	epochs = 8000, 
	batchSize = 256, 
	learningRate = .0002, 
	saveFreq = 100, 
	inputPath = "neurons/*", 
	loadNewData = True, 
	numGPUs = 1)

