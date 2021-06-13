import argparse
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

from mtcnn import MTCNN

PORTRAIT_PAINTING_UNCROPPED = 'portrait_painting_uncropped'
PORTRAIT_PAINTING = 'portrait_painting'
CELEB_A_W = 178
CELEB_A_H = 218


## 0. query proper data
## 1. detect face
## 2. crop proper area (+ resize)
## 3. save cropped image

def main(impressionism_or_all, version):
    ##########
    ## 0. query proper data
    ##########
    path1 = os.path.join(PORTRAIT_PAINTING_UNCROPPED,impressionism_or_all)
    artist_list = os.listdir(path1)

    print('***************************')
    print('Start processing:')
    for artist in artist_list:
        path2 = os.path.join(path1, artist)
        title_list = os.listdir(path2)

        for title in title_list:
            path3 = os.path.join(path2, title)
            
            img = cv2.imread(path3)            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            ##########
            ## 1. detect face
            ##########
            detector = MTCNN()

            try:
                box = detector.detect_faces(img)[0]['box']
                x_0 = box[0]
                y_0 = box[1]
                w = box[2]
                h = box[3]
                y_adj = y_0 - 0.07*2*h

                y_max = img.shape[0]
                x_max = img.shape[1]

                x_top_left = max(int(x_0 - w/2), 0)
                y_top_left = max(int(y_adj - h/2), 0)
                x_bottom_right = min(int(x_0 + w + w/2), x_max)
                y_bottom_right = min(int(y_adj + h + h/2), y_max)

                # cv2.rectangle(img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 4) # green
                # cv2.rectangle(img, (x_top_left, y_top_left + h), (x_bottom_right, y_top_left + h), (0, 255, 255), 4) # cyan
                # cv2.rectangle(img, (x_0, y_0), (x_0 + w, y_0 + h), (255, 0, 0), 4) # red
                # cv2.rectangle(img, (x_0, int(y_0 + h/2)), (x_0 + w, int(y_0 + h/2)), (255, 127, 0), 4) # orange

                ##########
                ## 2. crop proper area (+ resize)
                ##########
                crop_img = img[y_top_left : y_bottom_right, x_top_left : x_bottom_right]
                if version == '0': # resize image for version '0'                
                    crop_img = cv2.resize(crop_img, dsize=(CELEB_A_W, CELEB_A_H), interpolation=cv2.INTER_AREA)            

            except:                
                if version == '0' or version == '1':
                    print(f'No face detected for {artist}_{title}. Omit the image')
                    continue
                elif version == '2':
                    print('No face detected. Use the entire image')
                    crop_img = img
                                
                        
            ##########
            ## 3. save cropped image
            ##########                    
            save_path = os.path.join(PORTRAIT_PAINTING + '_' + version, impressionism_or_all, artist)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_path, title), crop_img)
    
    print('done')

if __name__ == '__main__':    
    """
    data versions
    - 0 | omit image if face is not detected       | resize cropped image (218, 178, 3) (same as celeb_a image size)
    - 1 | omit image if face is not detected       | no resize
    - 2 | use entire image if face is not detected | no resize
    """
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--impress_or_all', type=str, default='impressionsism')    
    parser.add_argument('--version', type=str, default='0')
        
    args = parser.parse_args()

    with tf.device('CPU:' + '0'):            
        main(args.impress_or_all, args.version)
        # main('all', '0')
        # main('impressionism', '1')
        # main('all', '1')
        # main('impressionism', '2')
        # main('all', '2')



