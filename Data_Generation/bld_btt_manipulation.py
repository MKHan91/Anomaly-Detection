from glob import glob
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import scipy.io as sci
import os

parser = argparse.ArgumentParser(description='blade btt data classification')
parser.add_argument('--mode',      type=str,   help='All of about blade btt data', default='correlation')
parser.add_argument('--data_path', type=str,
                    default='D:\\Onepredict_MK\\12 GuardiOne Turbine BVMS Deep Learning Development\\01_About_data\\03_raw_data\\1차_BTT_data')
parser.add_argument('--save_path', type=str,
                    default='D:\\Onepredict_MK\\16 GuardiOne Turbine BVMS Deep Learning Development\\02_Training_Data\\Mat_file\\BTT_image_train2_aug_20191122')

args = parser.parse_args()

def flutter(data_path, save_path):
    for idx, filename_mat in enumerate(sorted(os.listdir(data_path))):
        filename = filename_mat.split('.')[0]

        BTT_data_path = os.path.join(data_path, filename_mat)
        BTT_mat_file = sci.loadmat(BTT_data_path)
        BTT_mat = BTT_mat_file['tip_timing']

        # 1 BLADE
        BTT_mat_amp10 = BTT_mat
        mean_shift_amp10 = (BTT_mat_amp10[:, 0] - np.mean(BTT_mat_amp10[:, 0])) * 1.1
        BTT_mat_1B_amp10 = np.mean(BTT_mat_amp10[:, 0]) + mean_shift_amp10
        BTT_mat_amp10[:, 0] = BTT_mat_1B_amp10
        sci.savemat(save_path + '\\Amp_10' + '\\{}_flutter.mat'.format(filename), {'flutter_tip_timing': BTT_mat_amp10})

        BTT_mat_amp20 = BTT_mat
        mean_shift_amp20 = (BTT_mat_amp20[:, 0] - np.mean(BTT_mat_amp20[:, 0])) * 1.2
        BTT_mat_1B_amp20 = np.mean(BTT_mat_amp20[:, 0]) + mean_shift_amp20
        BTT_mat_amp20[:, 0] = BTT_mat_1B_amp20
        sci.savemat(save_path + '\\Amp_20' + '\\{}_flutter.mat'.format(filename), {'flutter_tip_timing': BTT_mat_amp20})

        BTT_mat_amp30 = BTT_mat
        mean_shift_amp30 = (BTT_mat_amp30[:, 0] - np.mean(BTT_mat_amp30[:, 0])) * 1.3
        BTT_mat_1B_amp30 = np.mean(BTT_mat_amp30[:, 0]) + mean_shift_amp30
        BTT_mat_amp30[:, 0] = BTT_mat_1B_amp30
        sci.savemat(save_path + '\\Amp_30' + '\\{}_flutter.mat'.format(filename), {'flutter_tip_timing': BTT_mat_amp30})

        BTT_mat_amp40 = BTT_mat
        mean_shift_amp40 = (BTT_mat_amp40[:, 0] - np.mean(BTT_mat_amp40[:, 0])) * 1.4
        BTT_mat_1B_amp40 = np.mean(BTT_mat_amp40[:, 0]) + mean_shift_amp40
        BTT_mat_amp40[:, 0] = BTT_mat_1B_amp40
        sci.savemat(save_path + '\\Amp_40' + '\\{}_flutter.mat'.format(filename), {'flutter_tip_timing': BTT_mat_amp40})

        BTT_mat_amp50 = BTT_mat
        mean_shift_amp50 = (BTT_mat_amp50[:, 0] - np.mean(BTT_mat_amp50[:, 0])) * 1.5
        BTT_mat_1B_amp50 = np.mean(BTT_mat_amp50[:, 0]) + mean_shift_amp50
        BTT_mat_amp50[:, 0] = BTT_mat_1B_amp50
        sci.savemat(save_path + '\\Amp_50' + '\\{}_flutter.mat'.format(filename), {'flutter_tip_timing': BTT_mat_amp50})

def Aug_v2(data_path, save_path):

    data_npys = glob(data_path + '\\*.npy')
    for idx, data_npy in enumerate(sorted(data_npys)):
        if idx == 0:
            total_data = np.load(data_npy)
            print(total_data.shape)
        else:
            total_data = np.concatenate((total_data, np.load(data_npy)), axis=0)
            print(total_data.shape)

    np.save(save_path+'\\total_norm_test_data.npy', total_data)

def DataAugmentation(data_path, save_path):
    filepaths = glob(data_path + '\\*.mat')
    for file_num, file_path in enumerate(sorted(filepaths)):
        data_name = file_path.split('\\')[-1].split('.')[0]

        data = sci.loadmat(file_path)
        data = data['image_']

        split_data = np.split(data, 60)
        for rand_num in range(20):
            random.shuffle(split_data)
            rnd_split_data = np.vstack(split_data)
            sci.savemat(save_path + '\\{}_aug{:03d}'.format(data_name, rand_num), {'image_':rnd_split_data})
            print('current processing: {}/{}, {}/{}'.format(rand_num + 1, 20, file_num+1, len(filepaths)))

def inter_correlation(data_path, save_path):
    data_list = sorted(glob(data_path + '\\*.mat'))

    iprd_list = []
    iprd_sum_list = []
    B1_btt_set = np.zeros((60, 53, 1200, 1))
    for bld in range(60):
        for num, mat_data in enumerate(data_list):
            tip_timing = sci.loadmat(os.path.join(data_path, mat_data))
            tip_timing = tip_timing['tip_timing']
            B1_btt = tip_timing[:,bld:bld+1]
            B1_btt_set[bld, :num, :] = B1_btt

    for bld in range(60):
        for idx in range(B1_btt_set.shape[1]):
            iprd_list += [np.dot(B1_btt_set[bld, : idx, :, :], B1_btt_set[bld, : idx+1, :, :])]

    iprd_sum_list += [sum(iprd_list)]

if __name__ == '__main__':
    if args.mode == 'flutter':
        flutter(args.data_path, args.save_path)

    elif args.mode == 'btt_aug_v2':
        Aug_v2(args.data_path, args.save_path)

    elif args.mode == 'augmentation':
        DataAugmentation(args.data_path, args.save_path)

    elif args.mode == 'correlation':
        inter_correlation(args.data_path, args.save_path)