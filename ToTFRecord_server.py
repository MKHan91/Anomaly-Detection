from glob import glob
from PIL import Image
import Data_Generation
import numpy as np
import os
import scipy.io as sci
import tensorflow as tf

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def check_or_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_folder():
    lambda_fn = [lambda i=i: i+1 for i in range(60)]
    for i in lambda_fn:
        blade_type_path = os.path.join(BASE_PATH, 'Raw_state', '20191113_tfrecord', '{:02d}_BLADE'.format(i()))
        check_or_create_dir(dir=blade_type_path)

def find_minmax_value(data_path):
    max_element = []
    min_element = []
    aug_data_paths = glob(data_path + '/*.mat')

    for data_num, aug_data_path in enumerate(sorted(aug_data_paths)):
        aug_data = sci.loadmat(aug_data_path)
        aug_data = aug_data['image_']

        max_element += [aug_data.max()]
        min_element += [aug_data.min()]

        print('find min max value: {}/{}'.format(data_num+1, len(aug_data_paths)))
    return max(max_element), min(min_element)

def convert_TFRecord_aboutMATFILE(data_path):
    print('Start converting')
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    max_value, min_value = find_minmax_value(data_path)

    test_path = os.path.join(TEST_PATH, 'Mat_file', '20191208_data_testMK_v5')
    # aug_data_paths = glob(data_path + '/*.mat')
    aug_data_paths = sorted(glob(test_path + '/*.mat'))
    for data_num, aug_data_path in enumerate(sorted(aug_data_paths)):
        data_name = aug_data_path.split('/')[-1].split('.')[0]

        aug_data = sci.loadmat(aug_data_path)
        aug_data = aug_data['image_']
        # aug_data = np.float32(aug_data)
        # norm_aug_data = np.float32((aug_data[50:, :] - min_value) / (max_value - min_value))
        norm_aug_data = np.float32((aug_data - min_value) / (max_value - min_value))
        binary_norm_aug_data = norm_aug_data.tobytes()

        writer = tf.python_io.TFRecordWriter(path=os.path.join(TEST_PATH, 'Raw_state', '20191208_data_testMK_ver5', data_name+'.tfrecord')
                                             ,options=options)
        feature = {
            'Image': _bytes_feature(binary_norm_aug_data)
        }

        # Serialize to string and write to file.
        string_set = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(string_set.SerializeToString())

        print('{}/{}'.format(data_num + 1, len(aug_data_paths)))
        writer.close()

def TFRecord_acoustics():
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    norm_paths = sorted(glob(BASE_PATH + '/*.png'))
    for num, n_path in enumerate(norm_paths):
        data_name = n_path.split('/')[-1].split('.')[0]
        png = Image.open(n_path)
        png_array = np.asarray(png)[:, :, 0:3]
        binary_png = png_array.tobytes()

        writer = tf.python_io.TFRecordWriter(
            path=os.path.join('/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20200120_normal_tfrecord', data_name + '.tfrecord')
            , options=options)
        feature = {
            'Image': _bytes_feature(binary_png)
        }

        # Serialize to string and write to file.
        string_set = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(string_set.SerializeToString())

        print('{}/{}'.format(num + 1, len(norm_paths)))
        writer.close()
if __name__ == '__main__':
    BASE_PATH = Data_Generation.BASE_PATH
    TEST_PATH = Data_Generation.TEST_PATH

    if Data_Generation.args.mode == 'matlab_test':
        convert_TFRecord_aboutMATFILE(os.path.join(BASE_PATH, 'Mat_file', 'BTT_image_train2_20191122'))
    elif Data_Generation.args.mode == 'Acoustics':
        TFRecord_acoustics()
