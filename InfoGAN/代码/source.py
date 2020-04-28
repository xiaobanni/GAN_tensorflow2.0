import tensorflow as tf 
import numpy as np
import scipy.misc
import os
import imageio

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images =  tf.expand_dims(train_images,axis=3)
    train_images = tf.cast(train_images, tf.float32)

    BUFFER_SIZE=train_images.shape[0]
    train_images=(train_images)/255.0
    train_labels=tf.one_hot(train_labels,depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(64,drop_remainder=True)
    return train_dataset

def inverse_transform(images):
    return (images+1.0)/2.0


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_images(images, size, image_path):
    return imwrite(inverse_transform(images), size, image_path)


def imwrite(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')