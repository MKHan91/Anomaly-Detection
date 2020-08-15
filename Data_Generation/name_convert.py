from glob import glob
import os
import csv
import shutil
import argparse
import scipy.io as sci
import numpy as np

parser = argparse.ArgumentParser(description='blade btt data classification')
parser.add_argument('--mode',      type=str,   help='mode corresponding to time point', default='name_convert')
parser.add_argument('--data_path', type=str,
                    default='D:\\Onepredict_MK\\16 GuardiOne Turbine BVMS Deep Learning Development\\01_About_data\\02_Test_Data\\Mat_file\\20191206_data_test_freq_MK')

args = parser.parse_args()

if args.mode == 'name_convert':
    filepaths = glob(os.path.join(args.data_path, '*.mat'))
    for num, f_path in enumerate(filepaths):
        filenames = f_path.split('\\')
        filename = filenames[-1].split('_')
        sec = int(filename[-1].split('.')[0])

        del filename[-1]

        filename.insert(3, '{:03d}'.format(sec))
        new_filename = '_'.join(filename)
        os.rename(src=f_path, dst=os.path.join(args.data_path, new_filename+'.mat'))
        print('current processing: {}/{}'.format(num+1, len(filepaths)))

elif args.mode == 'npy':
    filepaths = glob(os.path.join(args.data_path, '*.mat'))
    save_path = 'D:\\Onepredict_MK\\16 GuardiOne Turbine BVMS Deep Learning Development\\02_Training_Data\\Mat_file\\20191125_BVMS_DL_normal_data'

    data_npy = np.empty(shape=(len(filepaths), 600, 600))
    num = 1
    for idx, filepath in enumerate(sorted(filepaths)):
        mat_file = sci.loadmat(filepath)
        if 'image_' in mat_file.keys():
            image_arr = np.float32(mat_file['image_'])
            for num in range(len(filepaths)):
                data_npy[num, :, :] = image_arr
            # for bld_num in range(image_arr.shape[0]):
            #     mat_data = image_arr[bld_num, :]
            #     BTT_1x1200 = BTT_1x1200.reshape(1, 1200)
            #     BTT_1x1200_npy[(60*idx + bld_num), :, :] = BTT_1x1200
        elif 'aug_data' in mat_file.keys():
            image_arr = np.float32(mat_file['image_'])
            for num in range(len(filepaths)):
                data_npy[num, :, :] = image_arr

    np.save(file=save_path+'\\BTT.npy', arr=data_npy)
    # np.load(file=save_path+'\\BTT.npy')
