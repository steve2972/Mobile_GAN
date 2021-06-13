import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import random

import time


class Dataset:
    def __init__(self, dataset_name='mnist'):
        ds_train, self.ds_test = tfds.load(dataset_name, split=["train", "test"], as_supervised=True, shuffle_files=False)

        self.data = {}
        if dataset_name == 'omniglot':
            self.image_size = [32,32]
        else:
            self.image_size = [28,28]

        def extraction(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            if self.image_size != [28,28]:
                image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, self.image_size)
            return image, label

        for image, label in ds_train.map(extraction):
            image = image.numpy()
            label = str(label.numpy())
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
        self.labels = list(self.data.keys())
        self.labels.sort()

    def shuffle_labels(self):
        np.random.shuffle(self.labels)

    def get_mini_dataset(self, label_idx, batch_size=20):
        #random_label = self.labels[random.randint(0, len(self.labels)-1)]
        label = self.labels[label_idx]
        indices = np.random.choice(len(self.data[label]), batch_size, replace=False)
        dataset = np.array(self.data[label])[indices].astype(np.float32)
        return dataset
        
    def get_test_dataset(self, label_idx, batch_size=20):
        #random_label = self.test_labels[random.randint(0, len(self.test_labels)-1)]
        label = self.labels[label_idx]
        indices = np.random.choice(len(self.test_data[label]), batch_size, replace=False)
        dataset = np.array(self.test_data[label])[indices].astype(np.float32)

        return dataset




def define_discriminator(in_shape=(28, 28, 1)):
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2,2), padding='same', input_shape=in_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(64, (3, 3), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(output_width, input_dim):
    n = int(output_width/4)
    n_nodes = 128 * n * n # foundation for 8x8 image
    model = keras.Sequential([
        layers.Dense(n_nodes, input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((n,n,128)),
        # Upsample to 16x16
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        # Upsample to 32x32
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7,7), activation='sigmoid', padding='same')
        
    ])
    
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = keras.Sequential([
        g_model, d_model
    ])
    opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

"""
    Define data generation functions
"""

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_real_samples(dataset, label_idx, n_samples=20):
    images = dataset.get_mini_dataset(label_idx, n_samples)  # in a numpy array format
    labels = np.ones((len(images), 1))
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(100).batch(n_samples) # data.shape = (n_samples, 28, 28, 1)
    return data

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X_gan = g_model(x_input) # generated fake image shape: (n_samples, 28, 28, 1)
    y = np.zeros((n_samples, 1))
    data = tf.data.Dataset.from_tensor_slices((X_gan, y))
    data = data.shuffle(100).batch(n_samples)    
    
    return data

def generate_train_samples(latent_dim, n_samples):
    X_gan = generate_latent_points(latent_dim, n_samples)
    y_gan = np.ones((n_samples, 1))

    data = tf.data.Dataset.from_tensor_slices((X_gan, y_gan))
    data = data.shuffle(100).batch(n_samples)    
    
    return data

"""
    Define testing functions
"""

def compare_plot(real_examples, fake_examples, epoch, n_samples):
    for i in range(2):
        for j in range(n_samples):
            plt.subplot(2, n_samples, 1 + (i*n_samples) + j)
            plt.axis('off')
            if i == 0:
                plt.imshow(real_examples[j, :, :, 0], cmap='gray_r')
            elif i == 1:
                plt.imshow(fake_examples[j, :, :, 0], cmap='gray_r')
    filename='img/compare_plot_e%03d.png'%(epoch+1)
    plt.savefig(filename)
    plt.close()
    
def summarize_performance(epoch, g_model, d_model, gan_model, dataset, latent_dim, n_samples, inner_loops=10, label_idx=0, shuffle=True):
    # Algorithm 2. FIGR Generation
    if shuffle: dataset.shuffle_labels()
    theta_g = g_model.get_weights()
    theta_d = d_model.get_weights()


    real_data = generate_real_samples(dataset, label_idx, n_samples)
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
    
    all_data = real_data.concatenate(fake_data).shuffle(100)

    d_model.fit(all_data, epochs=inner_loops, verbose=0)

    
    train_data = generate_train_samples(latent_dim, n_samples)
    gan_model.fit(train_data, epochs=inner_loops, verbose=0)
    
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
   
    loss_real, _ = d_model.evaluate(real_data, verbose=0)
    loss_fake, _ = d_model.evaluate(fake_data, verbose=0)
    
    print(">Round {}\n\t-Discriminator loss: {:.2f}\n\t-Generator loss: {:.2f}".format(epoch, loss_fake + loss_real, loss_fake))
    
    X_real = dataset.get_mini_dataset(label_idx, n_samples)
    x_input = generate_latent_points(latent_dim, n_samples)
    X_fake = g_model(x_input)

    g_model.set_weights(theta_g)
    d_model.set_weights(theta_d)
    

    compare_plot(X_real, X_fake, epoch, n_samples)
    filename_g='model/generator_model_%03d.h5' % (epoch+1)
    filename_d='model/discriminator_model_%03d.h5' % (epoch+1)
    g_model.save(filename_g)
    d_model.save(filename_d)

def test_model(g_model, d_model, gan_model, dataset, latent_dim, n_samples, inner_loops=10, label_idx=-1):
    start = time.time()
    theta_g = g_model.get_weights()
    theta_d = d_model.get_weights()


    real_data = generate_real_samples(dataset, label_idx, n_samples)
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
    
    all_data = real_data.concatenate(fake_data).shuffle(100)

    d_model.fit(all_data, epochs=inner_loops, verbose=0)

    
    train_data = generate_train_samples(latent_dim, n_samples)
    gan_model.fit(train_data, epochs=inner_loops, verbose=0)
    
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
        
    X_real = dataset.get_mini_dataset(label_idx, n_samples)
    x_input = generate_latent_points(latent_dim, n_samples)
    X_fake = g_model(x_input)

    g_model.set_weights(theta_g)
    d_model.set_weights(theta_d)
    
    end = time.time()
    print(f"Time elapsed: {end - start:.5f}")
    compare_plot(X_real, X_fake, -1, n_samples)
    return end - start

"""
    Define training functions
"""

def weight_difference(old_weights, weights):
    new_weights = list()
    for i in range(len(weights)):
        new_weights.append((old_weights[i] - weights[i]))
    return new_weights

def average_weights(weight_list):
    avg_weight = list()
    for weight in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(weight, axis=0)
        avg_weight.append(layer_mean)

    return avg_weight

def grad_dimension_match(grads, model):
    final_weights = list()
    j = 0
    for grad in grads:
        if grad.shape == model.trainable_weights[j].shape:
            j += 1
            final_weights.append(grad)
    return final_weights

def inner_loop(g_model, d_model, gan_model, dataset, n_samples, inner_loops, latent_dim, label_idx):
    # Sample a task from the training dataset 
    real_data = generate_real_samples(dataset, label_idx, n_samples)
    # Generate latent vectors -> generate fake images
    fake_data = generate_fake_samples(g_model, latent_dim, n_samples)
            
    all_data = real_data.concatenate(fake_data).shuffle(100)
    
    d_model.fit(all_data, epochs=inner_loops, verbose=0)
                    
    # Train the generator
    train_data = generate_train_samples(latent_dim, n_samples)
    gan_model.fit(train_data, epochs=inner_loops, verbose=0)

def reptile_train(g_model, d_model, gan_model, 
                  dataset, latent_dim, n_samples,
                  inner_loops = 10, n_epochs=200, n_tasks=501,
                  meta_step_size=0.00001, shuffle=True, random_label=False):
        
    # Algorithm 1. FIGR training
    
    opt_g = keras.optimizers.Adam(learning_rate=meta_step_size)
    opt_d = keras.optimizers.Adam(learning_rate=meta_step_size)

    g_grads_list = list()
    d_grads_list = list()
    
    # How many epochs the model should train for
    for i in range(n_epochs):
        # On how many labels/classes the model should be optimized on per epoch
        for j in range(n_tasks):
            # Make a copy of theta_generator and theta_discriminator
            theta_g = g_model.get_weights()
            theta_g_grads = g_model.trainable_weights
            theta_d = d_model.get_weights()
            theta_d_grads = d_model.trainable_weights

            label_idx = random.randint(0, len(dataset.labels)-1) if random_label==True else j
            # Train the GAN model & discriminator
            inner_loop(g_model, d_model, gan_model, dataset, n_samples, inner_loops, latent_dim, label_idx)
        
            # Get the newly trained model weights
            phi_g = g_model.trainable_weights
            phi_d = d_model.trainable_weights

            # Calculate weight difference between optimized weights and original weights
            g_grads = weight_difference(phi_g, theta_g_grads)
            d_grads = weight_difference(phi_d, theta_d_grads)
            #g_grads_list.append(g_grads)
            #d_grads_list.append(d_grads)

            # Set the model weights to the original theta
            #g_model.set_weights(theta_g)
            #d_model.set_weights(theta_d)

            opt_g.apply_gradients(zip(g_grads, g_model.trainable_weights))
            opt_d.apply_gradients(zip(d_grads, d_model.trainable_weights))

        #g_grads = average_weights(g_grads_list)
        #d_grads = average_weights(d_grads_list)

        #opt_g.apply_gradients(zip(g_grads, g_model.trainable_weights))
        #opt_d.apply_gradients(zip(d_grads, d_model.trainable_weights))

        if shuffle: dataset.shuffle_labels()

        if (i + 1) % 10 == 0:
            summarize_performance(i+1, g_model, d_model, gan_model, dataset, latent_dim, n_samples, inner_loops=inner_loops, label_idx=-1, shuffle=shuffle)

def main():
    # Tensorflow GPU settings
    gpu_num = 2
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)

    dataset = Dataset(dataset_name='mnist')

    latent_dim=100
    n_samples=5
    n_tasks=9
    epochs=200
    shuffle=False
    rl = False


    d_model = define_discriminator(in_shape=(28, 28, 1))
    d_model.load_weights('./model/discriminator_model_400.h5')
    g_model = define_generator(output_width=28, input_dim=latent_dim)
    g_model.load_weights('./model/generator_model_400.h5')

    gan_model = define_gan(g_model, d_model)

    times = list()
    for i in range(10):
        times.append(test_model(g_model, d_model, gan_model, dataset, latent_dim, n_samples))

    print(np.average(times[1:]))
    #reptile_train(g_model, d_model, gan_model, dataset, 
    #                latent_dim, n_samples, n_epochs=epochs, 
    #                n_tasks=n_tasks, shuffle=shuffle, random_label=rl)

if __name__ == '__main__':
  main()
