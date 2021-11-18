import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib as plt

from MyModel import *
import os


class Classifier:

    def __init__(self, data=None, model=None ):
        self.dataset = data
        self.model = MyModel(dim_hidden=(2,511),perceptrons_out=10)


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
        #mnist = mnist.cache()
        #shuffle, batch, prefetch

        #mnist = mnist.map(lambda img, target: (tf.math.ceil(img), target))

        digit = digit.shuffle(1350)
        digit = digit.batch(32)
        digit = digit.prefetch(20)
        #return preprocessed dataset
        return digit

    
   
    def train(self):

        self.path = "./dataset/train_data/"
        train_ds = tf.data.Dataset.from_generator(self.data_generator, (tf.float32, tf.uint8))
        self.path = "./dataset/test_data/"
        test_ds = tf.data.Dataset.from_generator(self.data_generator, (tf.float32, tf.uint8))

        train_dataset = train_ds.apply(self.prepare_digit_data)
        test_dataset = test_ds.apply(self.prepare_digit_data)


        train_dataset = train_dataset.take(100)
        test_dataset = test_dataset.take(100)


        tf.keras.backend.clear_session()

        # trainig model
        tr,te,te_acc = training_loop(self.model,train_dataset,test_dataset, num_epochs=5, learning_rate=0.01)

        # visualize 
       # visualize_learning(tr,te,te_acc)


    def classify(self, data):


        data = tf.convert_to_tensor(data, np.float32)
        data= tf.reshape(data, (1,784))
        #print(data.shape)
        prediction = self.model(data)
        return prediction



def visualize_learning(train_losses,test_losses,test_accuracies): 
        """
        Visualize accuracy and loss for training and test data.
        """
        plt.figure()
        line1, = plt.plot(train_losses)
        line2, = plt.plot(test_losses)
        line3, = plt.plot(test_accuracies)
        plt.xlabel("Training steps")
        plt.ylabel("Loss/Accuracy")
        plt.legend((line1,line2, line3),("training losses", "test losses", "test accuracy"))
        
        return plt.show()

## testing ##
myclassifier = Classifier()
myclassifier.train()

