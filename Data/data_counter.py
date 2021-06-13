import argparse
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf




def main(landscape_or_portrait):    
    impressionsim_or_all_list = ['impressionism', 'all']
    legend = dict()
    
    for impressionism_or_all in impressionsim_or_all_list:
        path1 = os.path.join(landscape_or_portrait, impressionism_or_all)
        if os.path.exists(path1):
            label_list = os.listdir(path1)
        else:
            break 

        for label in label_list:
            if label not in legend.keys():
                legend[label] = []

                path2 = os.path.join(path1, label)

                if os.path.exists(path2):
                    data_list = os.listdir(path2)            
                    legend[label].append(len(data_list))
                else:
                    legend[label].append(0)                                                    

        df = pd.DataFrame.from_dict(legend, orient='index', columns=['no_of_images'])        
        # print(df)
        df.to_csv(f'{landscape_or_portrait}_{impressionism_or_all}.csv', sep=',', index_label=['task'], index=True)
        print(f'Saved {path1}.csv')
        
        legend = dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--land_or_port', type=str, default='landscape')
    parser.add_argument('--version', type=str, default='0') # version for portrait
    
    args = parser.parse_args()
    if args.land_or_port == 'landscape':
        landscape_or_portrait = f'{args.land_or_port}_painting'
    elif args.land_or_port == 'portrait':
        landscape_or_portrait = f'{args.land_or_port}_painting_{args.version}'
    main(landscape_or_portrait) 







