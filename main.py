import tensorflow as tf
from tensorflow import keras

import numpy as np

import utils.modelsv2
import utils.models
import os
import argparse

from pathlib import Path
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu_num", dest="gpu_num", type=int, default=1)
    parser.add_argument("-ir", "--inner_rounds", dest="ir", type=int, default=5)
    parser.add_argument("-or", "--outer_rounds", dest="outer", type=int, default=1)
    
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=50)
    

    parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument("-ms", "--meta-step", dest="meta_step", type=float, default=2e-4)
    parser.add_argument("-s", "--shots", dest="shots", type=int, default=5)
    parser.add_argument("-t", "--tasks", dest="tasks", type=int, default=5)
    


    parser.add_argument("-bs", "--buffer-size", dest="buffer_size", type=int, default=256)

    parser.add_argument("-ds", "--dataset", dest="dataset", type=str, default='landscape')
    parser.add_argument("-style", "--style", dest="style", type=str, default='impressionism')
    


    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[args.gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu_num], True)

    autotune = tf.data.experimental.AUTOTUNE
    """
        IMPORT & PREPROCESS THE DATA
    """

    if args.style == 'impressionism':
        if args.dataset == 'landscape':
            photo_dir= '/home/aiot/data/MobileGAN/landscape/landscape_photos/train/'
            impressionism_dir = '/home/aiot/data/MobileGAN/landscape/landscape_painting/impressionism/train/'
            test_photo_dir = '/home/aiot/data/MobileGAN/landscape/landscape_photos/test/'
            test_impressionism_dir = '/home/aiot/data/MobileGAN/landscape/landscape_painting/impressionism/test/'
        elif args.dataset == 'portrait':
            photo_dir = '/home/aiot/data/MobileGAN/portrait/img_align_celeba/train/'
            impressionism_dir = '/home/aiot/data/MobileGAN/portrait/portrait_painting/impressionism/train/'
            test_photo_dir = '/home/aiot/data/MobileGAN/portrait/img_align_celeba/test/'
            test_impressionism_dir = '/home/aiot/data/MobileGAN/portrait/portrait_painting/impressionism/test/'
    elif args.style == 'all':
        if args.dataset == 'landscape':
            photo_dir= '/home/aiot/data/MobileGAN/landscape/landscape_photos/train/'
            impressionism_dir = '/home/aiot/data/MobileGAN/landscape/landscape_painting/all/train'
            test_photo_dir = '/home/aiot/data/MobileGAN/landscape/landscape_photos/test/'
            test_impressionism_dir = '/home/aiot/data/MobileGAN/landscape/landscape_painting/impressionism/test/'
        elif args.dataset == 'portrait':
            photo_dir = '/home/aiot/data/MobileGAN/portrait/img_align_celeba/train/'
            impressionism_dir = '/home/aiot/data/MobileGAN/portrait/portrait_painting/all/train'
            test_photo_dir = '/home/aiot/data/MobileGAN/portrait/img_align_celeba/test/'
            test_impressionism_dir = '/home/aiot/data/MobileGAN/portrait/portrait_painting/impressionism/test/'

    if args.dataset == 'landscape':
        input_size = (256, 256, 3)
        pfp_train = preprocess_function_photo_train
        pfa_train = preprocess_function_artist_train
        pfp_test = preprocess_function_photo_test
        pfa_test = preprocess_function_artist_test
    elif args.dataset == 'portrait':
        input_size = (220, 180, 3)
        pfp_train = preprocess_function_photo_train2
        pfa_train = preprocess_function_artist_train2
        pfp_test = preprocess_function_photo_test2
        pfa_test = preprocess_function_artist_test2


    ####################################################################################################
    ####################################################################################################

    print("Importing photo dataset")
    jpg_dirs = os.listdir(photo_dir)
    np.random.shuffle(jpg_dirs)
    jpg_dirs = np.random.choice(jpg_dirs, min(len(jpg_dirs), 5000), replace=False)

    filenames = tf.constant(jpg_dirs)
    photos = tf.data.Dataset.from_tensor_slices(photo_dir+filenames)
    photos = photos.map(pfp_train).batch(args.shots).prefetch(autotune)

    ####################################################################################################
    jpg_dirs_test = os.listdir(test_photo_dir)
    np.random.shuffle(jpg_dirs_test)
    jpg_dirs_test = np.random.choice(jpg_dirs_test, min(len(jpg_dirs_test), 5000), replace=False)

    test_filenames = tf.constant(jpg_dirs_test)
    test_photos = tf.data.Dataset.from_tensor_slices(test_photo_dir+test_filenames)
    test_photos = test_photos.map(pfp_test).prefetch(autotune).batch(1).take(args.shots)

    ####################################################################################################
    ####################################################################################################

    print("Importing artist dataset")
    artists = os.listdir(impressionism_dir)#[:5]
    np.random.shuffle(artists)
    paintings = list()
    artist_names = list()

    for artist in artists:
        jpg_dirs = os.listdir(impressionism_dir + artist)
        if len(jpg_dirs) >= args.shots:
            sampled_dirs = np.random.choice(jpg_dirs, len(jpg_dirs) // args.shots * args.shots, replace=False)
            for i in sampled_dirs:
                paintings.append(i)
                artist_names.append(artist)
    paintings = tf.constant(paintings)
    artist_names = tf.constant(artist_names)

    tasks = tf.data.Dataset.from_tensor_slices((impressionism_dir+artist_names+'/'+paintings, artist_names))
    tasks = tasks.map(pfa_train).batch(args.shots).prefetch(autotune)

    ####################################################################################################

    test_artists = os.listdir(test_impressionism_dir)
    np.random.shuffle(test_artists)
    test_paintings = list()
    test_artist_names = list()

    for artist in test_artists:
        jpg_dirs = os.listdir(test_impressionism_dir + artist)
        if len(jpg_dirs) >= args.shots:
            sampled_dirs = np.random.choice(jpg_dirs, len(jpg_dirs) // args.shots * args.shots, replace=False)
            for i in sampled_dirs:
                test_paintings.append(i)
                test_artist_names.append(artist)

    test_paintings = tf.constant(test_paintings)
    test_artist_names = tf.constant(test_artist_names)

    test_tasks = tf.data.Dataset.from_tensor_slices((test_impressionism_dir+test_artist_names+'/'+test_paintings, test_artist_names))
    test_tasks = test_tasks.map(pfa_test).prefetch(autotune).batch(1).take(args.shots)


    ####################################################################################################
    ####################################################################################################
    # Create path files
    plot_filepath = f'./images/{args.style}/{args.dataset}/{args.shots}shots_{args.ir}ir_{args.outer}or_{args.learning_rate}lr_{args.meta_step}ms'
    weight_filepath = f'./weights/{args.style}/{args.dataset}/{args.shots}shots_{args.ir}ir_{args.outer}or_{args.learning_rate}lr_{args.meta_step}ms'

    Path(plot_filepath).mkdir(parents=True, exist_ok=True)
    Path(weight_filepath).mkdir(parents=True, exist_ok=True)

    
    ####################################################################################################
    ####################################################################################################
    """
        DEFINE THE MODEL
    """
    # Get the generators
    gen_G = modelsv2.get_resnet_generator(name="generator_G", input_img_size=input_size)
    gen_F = modelsv2.get_resnet_generator(name="generator_F", input_img_size=input_size)

    # Get the discriminators
    disc_X = modelsv2.get_discriminator(name="discriminator_X", input_img_size=input_size)
    disc_Y = modelsv2.get_discriminator(name="discriminator_Y", input_img_size=input_size)


    # Create cycle gan model
    cycle_gan_model = modelsv2.CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
        meta_optimizer=keras.optimizers.SGD(learning_rate=args.meta_step)
    )

    # Create a meta-model for fine-tuned tasks
    cycle_gan_model_original = models.CycleGan(generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)
    cycle_gan_model_original.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    """
        START TRAINING
    """

    # Create meta-test task
    plotter = GANMonitor(test_photos, test_tasks, cycle_gan_model_original, plot_filepath, weight_filepath, inner_rounds=args.ir)

    # Training task
    train_data = tf.data.Dataset.zip((photos, tasks)).repeat(args.outer)
    
    # Train the model with GANMonitor callbacks
    cycle_gan_model.fit(train_data, epochs=args.epochs, verbose=1, callbacks=[plotter])

    

if __name__ == '__main__':
    main()
