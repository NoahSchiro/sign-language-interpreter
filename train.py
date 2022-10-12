import matplotlib 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf

# Some constant data
TRAINING_DIR = 'dataset/asl_alphabet_train'
TEST_DIR = 'dataset/asl_alphabet_test'
IMAGE_SIZE = 64
BATCH_SIZE = 64

# Helper function that will convert a "normal"
# image to an image which is grayscale and has edge detection
def edged_image(image):
    edged = cv2.Canny(np.uint8(image), 100, 200)
    color_edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    return np.float16(color_edged)

# Generate batches of tensor image data with real-time data augmentation.
data_generator = ImageDataGenerator(samplewise_center=True, 			# Set each sample mean to 0. 
									samplewise_std_normalization=True,  # Divide each input by its std. 
									validation_split=0.1				# Fraction of images reserved for validation
)

# Takes the path to a directory & generates batches of augmented data.
# Returns a DirectoryIterator object. Is a memory efficient way of handling our data
train_generator = data_generator.flow_from_directory(
	TRAINING_DIR, 							# Data from where?
	target_size=(IMAGE_SIZE, IMAGE_SIZE), 
	shuffle=True, 
	class_mode='categorical',
	batch_size=BATCH_SIZE, 
	subset="training"
)

# Takes the path to a directory & generates batches of augmented data.
# Returns a DirectoryIterator object. Is a memory efficient way of handling our data
validation_generator = data_generator.flow_from_directory(
	TRAINING_DIR,  							# Data from where?
	target_size=(IMAGE_SIZE, IMAGE_SIZE),
	shuffle=True,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset="validation"
)

# Define model
# This model has ~2 million trainable parameters
model = models.Sequential()

# Input layer (CNN)
model.add(layers.Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
# Hidden layers
model.add(layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, kernel_size=4, strides=1, activation='relu'))
model.add(layers.Conv2D(128, kernel_size=4, strides=2, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(256, kernel_size=4, strides=1, activation='relu'))
model.add(layers.Conv2D(256, kernel_size=4, strides=2, activation='relu'))

# Dense model
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(29, activation='softmax'))

# Use stochastic gradient decent with particular parameters (very slow training)
opt = optimizers.SGD(
	learning_rate=0.0001,
	momentum=0.9
)

# Compile model with adam optimizerer
model.compile(
	optimizer=opt,
    loss='categorical_crossentropy',	# because we are categorizing
    metrics=['accuracy'])				# We want to measure accuracy primarily

# Train model and track it's history
model.fit_generator(
	train_generator, 
	epochs=24, 
	validation_data=validation_generator)

MODEL_NAME = 'models/asl_classifier.h5'
model.save(MODEL_NAME)