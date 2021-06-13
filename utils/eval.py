#! -*- coding: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>
"""
metrics IS, FID, NDB and JSD.
"""

import os
import json
import ntpath
import numpy as np

import models
from util import util
import argparse
from itertools import combinations
from tqdm import tqdm
import data_ios

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_list', type=str, help='predict file list path')
parser.add_argument('--gt_list', type=str, help='real file list path')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gpu_id', type=str, default='0',
                    help='default is 0th GPU')
parser.add_argument('--resize', type=int, default=128,
                    help='128 for NDB and JSD; 299 for FID and IS')
parser.add_argument('--num_bins', default=100, help='used in NDB and JSD')
parser.add_argument('--dataset', type=str, help='dataset to be tested')
parser.add_argument('--metric', type=str)
args = parser.parse_args()


def print_eval_log(opt):
    message = ''
    message += '----------------- Eval ------------------\n'
    for k, v in sorted(opt.items()):
        message += '{:>20}: {:<10}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)


if __name__ == '__main__':
    use_cuda = args.gpu_id != ''
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    batch_size = args.batch_size
    metric_mode = args.metric
    if metric_mode == 'fid':
        pred_list, gt_list = [], []
        # with open(args.pred_list, 'r') as fin_pred:
        #     pred_list = [line.strip() for line in fin_pred]
        pred_list = [args.pred_list+dir for dir in os.listdir(args.pred_list)]
        gt_list = [args.gt_list+dir for dir in os.listdir(args.gt_list)]

        final_score = 0.0
        from scores.fid_scores import cal_fid as fid_score
        real_data_generator = data_ios.data_prepare_fid_is(
            gt_list, batch_size, args.resize, use_cuda)
        fake_data_generator = data_ios.data_prepare_fid_is(
            pred_list, batch_size, args.resize, use_cuda)
        dims = 2048
        final_score = fid_score(real_data_generator,
                                fake_data_generator, dims, use_cuda)

        logs = {'num_of_files': len(pred_list),
                'metric_mode': metric_mode,
                'final_score': final_score}
        print_eval_log(logs)

    elif metric_mode == 'lpips':
        # Initializing the model
        model = models.PerceptualLoss(model='net-lin', net='alex')
        dists = []
        pred_list = [args.pred_list+dir for dir in os.listdir(args.pred_list)]
        # f = list(combinations(pred_list, 2))
        for i in tqdm(range(len(pred_list)-1)):
            path0, path1 = pred_list[i], pred_list[i+1]
            img0 = util.im2tensor(util.load_image(os.path.join(path0)))
            img1 = util.im2tensor(util.load_image(os.path.join(path1)))

            img0 = img0.cuda()
            img1 = img1.cuda()

            # Compute distance
            dist01 = model.forward(img0, img1).data.cpu().squeeze().numpy()
            # print('Distance: %.4f' % dist01)
            dists.append(dist01)

        print('Average distance: %.4f' % (sum(dists)/len(dists)))
        print('Standard deviation:', np.array(dists).std())
