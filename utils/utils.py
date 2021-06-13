import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

import pickle


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, input_images, artist, cyclegan, path, weight_path, num_img=4, inner_rounds=5):
        self.num_img = num_img
        self.images = input_images
        self.artist = artist
        self.path = path
        self.weight_path = weight_path
        self.cyclegan = cyclegan
        self.ir = inner_rounds

    def on_epoch_end(self, epoch, logs=None):
        self.cyclegan.set_weights(self.model.get_weights())
        fine_tune_task = tf.data.Dataset.zip((self.images, self.artist))
        self.cyclegan.fit(fine_tune_task, epochs=self.ir, verbose=0)

        _, ax = plt.subplots(4, 4, figsize=(12, 12))
        paintings = [img for img in self.artist.take(self.num_img)]
        for i, img in enumerate(self.images.take(self.num_img)):
            prediction = self.cyclegan.gen_G(img)
            reconstructed = self.cyclegan.gen_F(prediction)
            
            prediction = prediction[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            reconstructed = reconstructed[0].numpy()
            reconstructed = (reconstructed * 127.5 + 127.5).astype(np.uint8)

            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
            painting = (paintings[i][0] * 127.5 + 127.5).numpy().astype(np.uint8)
            
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 2].imshow(reconstructed)
            ax[i, 3].imshow(painting)

            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 2].set_title("Reconstructed image")
            ax[i, 3].set_title("Target artist")

            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")
        
        plt.savefig("{}/generated_img_{}.png".format(self.path, epoch + 1))
        plt.show()
        plt.close()

        with open(f"{self.weight_path}/weights_epoch{epoch+1}.pkl", 'wb') as f:
            pickle.dump(self.model.get_weights(), f)


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def preprocess_function_photo_train(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def preprocess_function_artist_train(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def preprocess_function_photo_test(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def preprocess_function_artist_test(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


###################################################
def preprocess_function_photo_train2(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [220, 180])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def preprocess_function_artist_train2(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [220, 180])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def preprocess_function_photo_test2(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [220, 180])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

def preprocess_function_artist_test2(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [220, 180])
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0

# Define the loss function for the generators
def generator_loss_fn(fake):
    # Loss function for evaluating adversarial loss
    adv_loss_fn = keras.losses.MeanSquaredError()
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    adv_loss_fn = keras.losses.MeanSquaredError()
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def plot_graph(artist_name, images, model, artist, num_img=4):
    _, ax = plt.subplots(4, 4, figsize=(12, 12))
    paintings = [img for img in artist.take(num_img)]
    for i, img in enumerate(images.take(num_img)):
        prediction = model.gen_G(img)
        reconstructed = model.gen_F(prediction)
        
        prediction = prediction[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        reconstructed = reconstructed[0].numpy()
        reconstructed = (reconstructed * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
        painting = (paintings[i][0] * 127.5 + 127.5).numpy().astype(np.uint8)
        
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 2].imshow(reconstructed)
        ax[i, 3].imshow(painting)
        ax[i, 0].set_title("Input image")
        ax[i, 1].set_title("Translated image")
        ax[i, 2].set_title("Reconstructed image")
        ax[i, 3].set_title("Target artist")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 2].axis("off")
        ax[i, 3].axis("off")
        
    plt.savefig("{}_generated_img.png".format(artist_name))
    plt.show()
    plt.close()

def plot_pictures(artist_name, images, model, num_img=5):
    for i, img in enumerate(images.take(num_img)):
        prediction = model.gen_G(img)
        prediction = prediction[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

        plt.imshow(prediction)
        plt.axis("off")
        plt.show()
        plt.savefig(f"Generated/{artist_name}/{i}.png", bbox_inches='tight')
        plt.close()


@tf.function
def grads_difference(theta, phi):
    grads = list()
    for x, y in zip(theta, phi):
        grads.append(tf.math.subtract(x, y))
    return grads

def calculate_fid(generated, target, model):
    # Return feature map of images using InceptionV3
    act1 = model.predict(generated)
    act2 = model.predict(target)

    # Calculate the mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate squared difference of means
    ssdif = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdif + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid
