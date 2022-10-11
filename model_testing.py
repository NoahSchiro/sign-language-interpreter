import matplotlib 
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Constants
MODEL_NAME = 'models/asl_classifier.h5'
TRAINING_DIR = 'dataset/asl_alphabet_train'
TEST_DIR = 'dataset/asl_alphabet_test'
IMAGE_SIZE = 64
BATCH_SIZE = 64

# Load in the model we trained
model = load_model(MODEL_NAME)

# Get all of the possible classes
classes = os.listdir('dataset/asl_alphabet_train')
for category in classes:
    image_location = 'dataset/asl_alphabet_test/{}/{}_test.jpg'.format(category, category)
    img = cv2.imread(image_location)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    plt.figure()
    plt.imshow(img)
    img = np.array(img) / 255.
    img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    #img = img.standardize(img)
    prediction = np.array(model.predict(img))
    print('{} - {}'.format(category, classes[prediction.argmax()]))

test_data_generator = ImageDataGenerator(samplewise_center=True, 
                                         samplewise_std_normalization=True)

test_generator = test_data_generator.flow_from_directory(
    TEST_DIR, 
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode='categorical'
)

metrics = model.evaluate(test_generator)
print(metrics)