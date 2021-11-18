import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from MyModel import *
import os


class Classifier:

    def data_generator(self):

        files = os.listdir(self.path)
        for fileName in files:
            data = np.loadtxt(self.path + fileName)

            label = fileName.split("_")[3][:-4]

            if label == '':
                continue

            label = int(label)
            yield data, label


    def prepare_digit_data(self, digit):
        #flatten the images into vectors
        digit = digit.map(lambda img, target: (tf.reshape(img, (-1,)), target))
        #create one-hot targets
        digit = digit.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
        #cache this progress in memory, as there is no need to redo it; it is deterministic after all
        digit = digit.cache()
        #shuffle, batch, prefetch

        #mnist = mnist.map(lambda img, target: (tf.math.ceil(img), target))

        digit = digit.shuffle(1350)
        digit = digit.batch(32)
        digit = digit.prefetch(20)
        #return preprocessed dataset
        return digit


    def __init__(self):
        
        self.path = "./dataset/train_data/"
        train_ds = tf.data.Dataset.from_generator(self.data_generator, (tf.float32, tf.uint8))
        self.path = "./dataset/test_data/"
        test_ds = tf.data.Dataset.from_generator(self.data_generator, (tf.float32, tf.uint8))

        train_dataset = train_ds.apply(self.prepare_digit_data)
        test_dataset = test_ds.apply(self.prepare_digit_data)

        #train_dataset = train_dataset.take(1350)
        #test_dataset = test_dataset.take(180)

        # #
        # ### Hyperparameters
        num_epochs = 30
        learning_rate = 0.01

        # Initialize the model.
        self.model = MyModel()
        # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
        cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
        # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Initialize lists for later visualization.
        train_losses = []

        test_losses = []
        test_accuracies = []

        #testing once before we begin
        test_loss, test_accuracy = test(self.model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


        # #check how model performs on train data once before we begin
        train_loss, _ = test(self.model, train_dataset, cross_entropy_loss)
        train_losses.append(train_loss)
        #
        # We train for num_epochs epochs.
        for epoch in range(num_epochs):
            print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

            #training (and checking in with training)
            epoch_loss_agg = []
            for input,target in train_dataset:
                train_loss = train_step(self.model, input, target, cross_entropy_loss, optimizer)
                epoch_loss_agg.append(train_loss)

            #track training loss
            train_losses.append(tf.reduce_mean(epoch_loss_agg))

            #testing, so we can track accuracy and test loss
            test_loss, test_accuracy = test(self.model, test_dataset, cross_entropy_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

         # Plot results
        plt.suptitle("Accuracy and loss for training and test data")
        x = np.arange(0, len(train_losses))

        # First subplot
        plt.subplot(121)
        plt.plot(x, test_accuracies, 'g')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        # Second subplot
        plt.subplot(122)
        plt.plot(x, train_losses, 'r', label="Train")
        plt.plot(x, test_losses, 'b', label= "Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        # Format
        plt.tight_layout()

        # Save and display
        plt.savefig("result.png")
        #plt.show()

    def classify(self, data):

        data = tf.convert_to_tensor(data, np.float32)
        data= tf.reshape(data, (1,784))
        #print(data.shape)
        prediction = self.model(data)
        return prediction
