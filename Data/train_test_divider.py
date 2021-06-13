import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main(path):       
       
   label_list = os.listdir(path)
   label_list = [f for f in label_list if f.endswith('.jpg')]
        
   split_point = int(len(label_list)/10)
   test_list = np.random.choice(label_list, split_point, replace=False)
   train_list = [f for f in label_list if f not in test_list]

   print(len(test_list))
   print(len(train_list))
        
   for image in test_list:
        source = os.path.join(path, image)
        target = os.path.join(path, 'test', image)

        shutil.move(source, target)
   
   for image in train_list:
        source = os.path.join(path, image)
        target = os.path.join(path, 'train', image)

        shutil.move(source, target)
   
   print('done')

if __name__ == '__main__':
   #path = '/home/aiot/data/MobileGAN/landscape/landscape_photos'
   path = '/home/aiot/data/MobileGAN/portrait/img_align_celeba'
   main(path)







