import tensorflow as tf
from tensorflow.keras import layers

def define_generator(output_width=28, latent_dim=100):
    n = int(output_width / 4)

    model = tf.keras.Sequential()
    model.add(layers.Dense(n*n*256, use_bias=False, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((n, n, 256)))
    assert model.output_shape == (None, n, n, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, n, n, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, n*2, n*2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, n*4, n*4, 1)

    return model

def define_discriminator(input_shape=(28,28,1)):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = tf.keras.Sequential([
        g_model, d_model
    ])
    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1 = 0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, batch_size, latent_dim, g_model, d_model, g_opt, d_opt):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = d_model(images, training=True)
      fake_output = d_model(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

    g_opt.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
    d_opt.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))