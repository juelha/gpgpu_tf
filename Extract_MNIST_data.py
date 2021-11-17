import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np
import cv2

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # required for plotting
# import matplotlib.pyplot as plt


def main():

    train_path = "./train_data/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    test_path = "./test_data/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    mnist, info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)
    print(info)

    train_ds ,test_ds = mnist
    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)


    train_dataset = train_dataset.take(60000)
    test_dataset = test_dataset.take(10000)

    datasets = [train_dataset, test_dataset]
    paths = [train_path, test_path]
    for i, dataset in enumerate(datasets):

        for j, elem in enumerate(dataset):
            img, label = elem

            # Remove dim for batching
            img = img[0]
            label = label[0]

            # Erosion filter
            img = img.numpy()
            kernel = np.ones((3,3), np.uint8)
            img_erosioned = cv2.erode(img, kernel , iterations=1)

            # Store in file
            fileName = paths[i] + f"DataNo_{j}_Label_{label}.txt"
            np.savetxt(fileName, img_erosioned, fmt='%i')

            # plt.gray()
            #
            # img_erosioned = np.loadtxt(fileName)
            # plt.imshow(img_erosioned)
            #
            # plt.show()
            # exit()

def prepare_mnist_data(mnist):
    #convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    #sloppy input normalization, just bringing image values from range [0, 255] to [0, 1]
    mnist = mnist.map(lambda img, target: ((img/255), target))

    # Binarize
    mnist = mnist.map(lambda img, target: (tf.math.ceil(img), target))

    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(1)
    mnist = mnist.prefetch(20)

    return mnist


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
