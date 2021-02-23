# python 3.7.3
# tensorflow 2.4.1
# matplotlib 3.3.3
# numpy 1.19.5
# opencv-python 4.5.1
import time
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2

fout = open('test.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
print("[TIMER] Process Time:", now, file = fout, flush = True)

# File location to save to or load from
MODEL_SAVE_PATH = './FashionMNIST_net.pth'
TRAIN_EPOCHS = 20
SAVE_EPOCHS = False
SAVE_LAST = False
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print('[INFO] GPU is detected.')
    print('[INFO] GPU is detected.', file = fout, flush = True)
else:
    print('[INFO] GPU not detected.')
    print('[INFO] GPU not detected.', file = fout, flush = True)
print('[INFO] Done importing packages.')
print('[INFO] Done importing packages.', file = fout, flush = True)

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        # For Conv2D: Outgoing Layers, Frame size.  Everything else needs a keyword.
        self.model.add(layers.Conv2D(6, 5, input_shape = input_shape, activation = 'relu'))
        # For MaxPooling2D, default strides is equal to pool_size.  Batch and layers are assumed to match whatever comes in.
        self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Conv2D(12, 5, activation = 'relu'))

        self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Flatten())
        # Now, we flatten to one dimension, so we go to just length 400.

        self.model.add(layers.Dense(120, activation = 'relu'))

        self.model.add(layers.Dense(60, activation = 'relu'))

        self.model.add(layers.Dense(40, activation = 'relu'))

        self.model.add(layers.Dense(10))
        # Now we're at length 10, which is our number of classes.
        self.optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
        self.loss = losses.MeanSquaredError()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
        print(summaryStr, file=fout)

print("[INFO] Loading Traning and Test Datasets.")
print("[INFO] Loading Traning and Test Datasets.", file=fout)

# Get the FashionMNIST Dataset
((trainX, trainY), (testX, testY)) = datasets.fashion_mnist.load_data()
# Convert from integers 0-255 to decimals 0-1.
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Convert labels from integers to vectors.
lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

net = Net((28, 28, 1))

print(net)   # prints to both console and file

results = net.model.fit(trainX, trainY, validation_data=(testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1)

plt.figure()
plt.plot(np.arange(0, 20), results.history['loss'])
plt.plot(np.arange(0, 20), results.history['val_loss'])
plt.plot(np.arange(0, 20), results.history['accuracy'])
plt.plot(np.arange(0, 20), results.history['val_accuracy'])
plt.show()
