from PIL import Image
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import Data_Generation
import numpy as np
import os
import scipy.io as sci
import tensorflow as tf
import matplotlib.pyplot as plt

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

def convert_TFRecord(data_path):
    print('Start converting')
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    image_list = glob(data_path + '\\*.npy')
    for num, image_path in enumerate(sorted(image_list)):
        data_name = image_path.split('\\')[-1].split('.')[0]
        # writer = tf.python_io.TFRecordWriter(path=os.path.join(BASE_PATH, 'Raw_state', '20191125_image_tfrecord',
        #                                                        image_path.split('\\')[-1].split('.')[0] + '.tfrecord')
        #                                      ,options=options)
        aug_npy_data = np.load(image_path)
        aug_npy_data_norm = (aug_npy_data - aug_npy_data.min())/(aug_npy_data.max() - aug_npy_data.min())
        aug_npy_data_norm = np.float32(aug_npy_data_norm)

        for idx in range(aug_npy_data_norm.shape[0]):
            writer = tf.python_io.TFRecordWriter(path=os.path.join(BASE_PATH, 'Raw_state', '20191125_image_tfrecord',
                                                                   data_name+'_{:03d}'.format(idx+1) + '.tfrecord')
                                                 , options=options)

            partial_data = aug_npy_data_norm[idx, :, :]
            binary_partial_data = partial_data.tobytes()

            feature = {
                'Image': _bytes_feature(binary_partial_data)
            }

            # Serialize to string and write to file.
            string_set = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(string_set.SerializeToString())
            print('converting tfrecord: {}/{}'.format(num + 1, len(image_list)))

            writer.close()


        # image = Image.open(fp=image_path)
        # image_arr = image.convert('L')
        # image_arr = np.asarray(image)[:, :, 0:3]
        # binary_image = image_arr.tobytes()

        # feature = {
        #     'Image': _bytes_feature(binary_image)
        # }
        #
        # # Serialize to string and write to file.
        # string_set = tf.train.Example(features=tf.train.Features(feature=feature))
        # writer.write(string_set.SerializeToString())
        # print('converting tfrecord: {}/{}'.format(num+1, len(image_list)))
        #
        # writer.close()

    # for type in sorted(os.listdir(data_path)):
    #     BLADE_path = os.path.join(data_path, type)
    #     image_list = glob(BLADE_path+'\\*.png')
    #     for num, img_path in sorted(enumerate(image_list)):
    #         writer = tf.python_io.TFRecordWriter(path=os.path.join(BASE_PATH, 'Raw_state', 'Training_tfrecord_v3', type,
    #                                                                img_path.split('\\')[-1].split('.')[0]+'.tfrecord')
    #                                              ,options=options)
    #         image = Image.open(img_path)
    #         image_arr = np.asarray(image)[:, :, 0:3]
    #         image = Image.fromarray(image_arr)
    #         _binary_image = image.tobytes()
    #
    #         # feature = {
    #         #     'height': _int64_feature(image.size[0]),
    #         #     'width': _int64_feature(image.size[1]),
    #         #     'mean': _float_feature(np.asarray(image).mean()),
    #         #     'std': _float_feature(np.asarray(image).std()),
    #         #     'max': _int64_feature(np.asarray(image).max()),
    #         #     'min': _int64_feature(np.asarray(image).min()),
    #         #     'Image': _bytes_feature(_binary_image),
    #         #     'filename': _bytes_feature(str.encode(filename))
    #         # }
    #
    #         # feature = {
    #         #     'Image': _bytes_feature(tf.compat.as_bytes(image_array.tostring())),
    #         # }
    #
    #         feature = {
    #             'Image': _bytes_feature(_binary_image)
    #         }
    #
    #         # Serialize to string and write to file.
    #         string_set = tf.train.Example(features=tf.train.Features(feature=feature))
    #         writer.write(string_set.SerializeToString())
    #         print("BLADE_type: {}, Current Processing:{}/{}".format(type, num + 1, 1800))
    #
    #     writer.close()

def convert_TFRecord_test(data_path):
    print('Start converting')
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    total_norm_test_data = data_path + '\\all\\total_norm_test_data.npy'
    total_npy_data = np.load(total_norm_test_data)
    data_min = total_npy_data.min()
    data_max = total_npy_data.max()

    num = 12

    for idx, norm_test_data in enumerate(glob(data_path + '\\*.npy')):

        a = str(num)
        # norm_test_data = data_path + '\\total_norm_test_data.npy'

        npy_data = np.load(norm_test_data)
        npy_data_norm = (npy_data - data_min) / (data_max - data_min)
        npy_data_norm = np.float32(npy_data_norm)

        for index in range(np.shape(npy_data)[0]):
            npy_data_norm_part = npy_data_norm[index]
            binary_npy_data_norm = npy_data_norm_part.tobytes()

            writer = tf.python_io.TFRecordWriter(
                path=os.path.join(TEST_PATH, 'Raw_state', '20191124_data_test_norm_tfrecord', 'norm_data_test_%s_%s' %(a[0], a[1]), '{:03d}_{}'.format(idx+1, index)+'.tfrecord'), options=options)

            feature = {
                'Image': _bytes_feature(binary_npy_data_norm)
            }

            # Serialize to string and write to file.

            string_set = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(string_set.SerializeToString())
            writer.close()

        print('converting tfrecord: {}'.format(idx + 1))
        print('norm_data_test_%d' % (num))
        num = num + 2


    # image_list = glob(data_path + '\\*.npy')
    # for num, image_path in enumerate(sorted(image_list)):
    #     writer = tf.python_io.TFRecordWriter(path=os.path.join(TEST_PATH, 'Raw_state', '20191124_data_test_tfrecord','data_test_1.2',
    #                                                            image_path.split('\\')[-1].split('.')[0] + '.tfrecord') ,options=options)
    #
    #     image = Image.open(fp=image_path)
    #     image_arr = np.asarray(image)[:, :, 0:3]
    #     binary_image = image_arr.tobytes()
    #
    #     feature = {
    #         'Image': _bytes_feature(binary_image)
    #     }
    #
    #     # Serialize to string and write to file.
    #     string_set = tf.train.Example(features=tf.train.Features(feature=feature))
    #     writer.write(string_set.SerializeToString())
    #     print('converting tfrecord: {}/{}'.format(num+1, len(image_list)))
    #
    #     writer.close()

def find_minmax_value(data_path):
    max_element = []
    min_element = []
    aug_data_paths = glob(data_path + '\\*.mat')

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

    test_path = os.path.join(TEST_PATH, 'Mat_file', 'data_test_050bent2_191128', '05_2bent')
    # aug_data_paths = glob(data_path + '\\*.mat')
    aug_data_paths = sorted(glob(test_path + '/*.mat'))
    for data_num, aug_data_path in enumerate(sorted(aug_data_paths)):
        data_name = aug_data_path.split('\\')[-1].split('.')[0]

        aug_data = sci.loadmat(aug_data_path)
        aug_data = aug_data['image_']
        norm_aug_data = np.float32((aug_data - min_value) / (max_value - min_value))
        binary_norm_aug_data = norm_aug_data.tobytes()
        # binary_norm_aug_data = aug_data.tobytes()

        writer = tf.python_io.TFRecordWriter(path=os.path.join(TEST_PATH, 'Raw_state', '20191128_data_test_bent3_tfrecord_MK', '05_2bent', data_name+'.tfrecord')
                                             ,options=options)

        feature = {
            'Image': _bytes_feature(binary_norm_aug_data)
        }

        # Serialize to string and write to file.
        string_set = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(string_set.SerializeToString())

        print('{}/{}'.format(data_num + 1, len(aug_data_paths)))
        writer.close()

if __name__ == '__main__':
    BASE_PATH = Data_Generation.BASE_PATH
    TEST_PATH = Data_Generation.TEST_PATH

    if Data_Generation.args.mode == 'image_train':
        convert_TFRecord(os.path.join(BASE_PATH, 'Mat_file', '20191125_BVMS_DL_normal_data', '20191125_BTT_aug_image_train_npy'))
    elif Data_Generation.args.mode == 'image_test':
        convert_TFRecord_test(os.path.join(TEST_PATH, 'Mat_file', 'norm_data_test_191124'))
    elif Data_Generation.args.mode == 'matlab_test':
        convert_TFRecord_aboutMATFILE(os.path.join(BASE_PATH, 'Mat_file', 'BTT_image_train2_20191122'))
