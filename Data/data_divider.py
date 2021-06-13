import argparse
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

COMMON = '/home/aiot/data/'

WIKIART = COMMON + 'wikiart'
WIKIART_CSV = COMMON + 'wikiart_csv'

LANDSCAPE = COMMON + 'landscape'
CELEB_A = COMMON + 'celeb_a'
# MONET = COMMON + 'monet2photo'

LANDSCAPE_PAINTING = 'landscape_painting'
PORTRAIT_PAINTING_UNCROPPED = 'portrait_painting_uncropped'

# LANDSCAPE_PHOTO = 'landscape_photo'
# PORTRAIT_PHOTO = 'portrait_photo'

PORTRAIT = 6
LANDSCAPE_GENRE = 4

def mkdir(landscape_or_portrait, impressionism_or_all, artist):
    if landscape_or_portrait == 'landscape':
        copy_path = os.path.join(LANDSCAPE_PAINTING, impressionism_or_all, artist)        
                
    elif landscape_or_portrait == 'portrait':
        copy_path = os.path.join(PORTRAIT_PAINTING_UNCROPPED, impressionism_or_all, artist)

    if not os.path.exists(copy_path):
        os.makedirs(copy_path)
    
    return copy_path

def copy_image(elem, copy_path, title):
    source = os.path.join(WIKIART, elem)
    target = os.path.join(copy_path, title)        
    shutil.copy(source, target)

def main(landscape_or_portrait, impressionism_or_all):
    if landscape_or_portrait == 'landscape':
        genre = LANDSCAPE_GENRE
    elif landscape_or_portrait == 'portrait':
        genre = PORTRAIT

    df1 = pd.read_csv(os.path.join(WIKIART_CSV, 'genre_train.csv'), header=None)
    df2 = pd.read_csv(os.path.join(WIKIART_CSV, 'genre_val.csv'), header=None)
    df = pd.concat([df1, df2], axis=0)
    # print(df)
    l = list(df[0][df[1] == genre])
    print('***************************')
    print(f'Number of {landscape_or_portrait} paintings:', len(l))
    print('Start processing:')

    for elem in l:
        try:
            style, artist_title = elem.split('/')

            artist, title = artist_title.split('_', 1)
                       
            if impressionism_or_all == 'impressionism':
                if 'Impressionism' in style: # 'Impressionism' or 'Post_Impressionism'
                    copy_path = mkdir(landscape_or_portrait, impressionism_or_all, artist)
                    copy_image(elem, copy_path, title)
                    continue
            elif impressionism_or_all == 'all':
                copy_path = mkdir(landscape_or_portrait, impressionism_or_all, artist)
                copy_image(elem, copy_path, title)                    

        except:
            print(f'error for {elem}')
            continue        
    
    print(f'done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--land_or_port', type=str, default='landscape')
    parser.add_argument('--impress_or_all', type=str, default='impressionsism')        
        
    args = parser.parse_args()

    main(args.land_or_port, args.impress_or_all)
    

