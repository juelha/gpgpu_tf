import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from MyModel import *
import os


class Classifier:


    def __init__(self, model=None, train_ds=None, test_ds=None):
        """
        
        """
        #self.model = model
        
        self.model = MyModel(dim_hidden=(2,511),perceptrons_out=10)

        self.train(num_epochs=30, learning_rate=0.01)
        

    def data_generator(self):

        files = os.listdir(self.path)
        for fileName in files:
            data = np.loadtxt(self.path + fileName)

            label = fileName.split("_")[3][:-4]

            if label == '':
                continue

            label = int(label)
            yield data, label


    def pipeline(self, digit):
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

    def visualize(self,train_losses,test_losses,test_accuracies):
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



    def train(self, num_epochs, learning_rate):
        
        # get data
        self.path = "./dataset/train_data/"
        train_ds = tf.data.Dataset.from_generator(self.data_generator, (tf.float32, tf.uint8))
        self.path = "./dataset/test_data/"
        test_ds = tf.data.Dataset.from_generator(self.data_generator, (tf.float32, tf.uint8))


        # preprocessing & preparing of data
        train_dataset = train_ds.apply(self.pipeline)
        test_dataset = test_ds.apply(self.pipeline)


        tf.keras.backend.clear_session()

        # Initialize the model 
       # self.model = MyModel(dim_hidden=(2,511),perceptrons_out=10)

        # Train the model
        self.model.training_loop(train_dataset,test_dataset, num_epochs, learning_rate)




## testing ##
#myclassifier = Classifier(MyModel(dim_hidden=(2,511),perceptrons_out=10))

#myclassifier.train(num_epochs=30, learning_rate=0.01)


