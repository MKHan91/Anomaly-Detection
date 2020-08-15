from PIL import Image
from glob import glob
from pylab import *
import matplotlib.pyplot as plt
import scipy.io as sci
import Data_Generation
import pandas as pd
import numpy as np
import shutil
import os


def check_or_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def Convert(data_path):
    data_trains = glob(os.path.join(data_path, 'Mat_file', '20191120_BVMS_DL_normal_data', '20191124_BTT_aug_image_train', '*.mat'))
    save_path = os.path.join(data_path, 'Mat_file', '20191120_BVMS_DL_normal_data', '20191124_BTT_image_file')

    # maps = [m for m in cm.datad if not m.endswith("_r")]
    # maps.sort()
    for num, matfiles in enumerate(sorted(data_trains)):
        filename = matfiles.split('\\')[-1].split('.')[0]

        mat_file = sci.loadmat(matfiles)
        if 'image_' in mat_file.keys():
            image_arr = np.float32(mat_file['image_'])
            # image_arr_norm = image_arr / image_arr.max()

            plt.imsave(fname=save_path + '\\{}.png'.format(filename), arr=image_arr, cmap='gray')
            print('current processing: {}/{}'.format(num + 1, len(data_trains)))
        else:
            image_arr = np.float32(mat_file['aug_data'])
            # image_arr_norm = image_arr / image_arr.max()

            plt.imsave(fname=save_path + '\\{}.png'.format(filename), arr=image_arr, cmap='gray')
            print('current processing: {}/{}'.format(num + 1, len(data_trains)))
        # for i, m in enumerate(maps):
        #     plt.imshow(image_arr, cmap=get_cmap(m))
        #     plt.show()
        #     plt.imsave(fname=save_path+'\\{}_{}.png'.format(filename, i+1), arr=image_arr, cmap=get_cmap(m))
        #     print('current processing: {}/{}'.format(i + 1, len(maps)))

def Convert_test(data_path):
    data_trains = glob(os.path.join(data_path, 'Mat_file', 'data_test_191124', 'data_test_1.8', '*.mat'))
    save_path = os.path.join(data_path, 'Raw_state', '20191124_data_test_image', 'data_test_1.8')

    for num, matfiles in enumerate(sorted(data_trains)):
        filename = matfiles.split('\\')[-1].split('.')[0]

        mat_file = sci.loadmat(matfiles)
        if 'image_' in mat_file.keys():
            image_arr = np.float32(mat_file['image_'])
            image_arr_norm = image_arr / image_arr.max()

            plt.imsave(fname=save_path + '\\{}.png'.format(filename), arr=image_arr_norm, cmap='gray')
            print('current processing: {}/{}'.format(num + 1, len(data_trains)))
        else:
            image_arr = np.float32(mat_file['aug_data'])
            image_arr_norm = image_arr / image_arr.max()

            plt.imsave(fname=save_path + '\\{}.png'.format(filename), arr=image_arr_norm, cmap='gray')
            print('current processing: {}/{}'.format(num + 1, len(data_trains)))

if __name__ == '__main__':
    args = Data_Generation.args
    BASE_PATH = Data_Generation.BASE_PATH
    TEST_PATH = Data_Generation.TEST_PATH

    if args.mode == 'image_train':
        Convert(BASE_PATH)
    elif args.mode == 'image_test':
        Convert_test(TEST_PATH)
