from glob import glob
import os
import tensorflow as tf

TFData = tf.data
class AutoencoderDataloader(object):
    def __init__(self, args):
        self.args = args

        if args.mode == 'train':
            Files, TFR_path = self.read_tfrecord(args.train_data_path)
            self.training_num_samples = len(glob(TFR_path))

            tfrecord_dataset = TFData.TFRecordDataset(filenames=Files, compression_type='GZIP', num_parallel_reads=args.num_threads)

            """Buffer size provides a way to tune the performance of your input pipeline
            Low buffer size causes not be shuffled in the right way"""
            tfrecord_dataset = tfrecord_dataset.repeat(count=self.training_num_samples)
            """ Repeats this dataset 8 times == 8 epoch 동안 같은 데이터가 추출됨."""
            tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size=self.training_num_samples)
            tfrecord_dataset = tfrecord_dataset.map(self._parse_function, num_parallel_calls=args.num_threads)

            tfrecord_dataset = tfrecord_dataset.batch(args.batch_size)
            tfrecord_dataset = tfrecord_dataset.prefetch(args.batch_size)

            iterator = tfrecord_dataset.make_initializable_iterator()
            self.iter_init_op = iterator.initializer
            self.model_input = iterator.get_next()
            

        else:
            test_Files, test_TFR_path = self.read_tfrecord([args.test_data_path, args.train_data_path])
            self.test_num_samples = len(glob(test_TFR_path))

            self.test_tfrecord_dataset = TFData.TFRecordDataset(filenames=test_Files[0], compression_type='GZIP',num_parallel_reads=args.num_threads)
            self.test_tfrecord_dataset = self.test_tfrecord_dataset.map(self._parse_function, num_parallel_calls=args.num_threads)
            self.test_tfrecord_dataset = self.test_tfrecord_dataset.batch(args.test_batch_size)
            self.test_tfrecord_dataset = self.test_tfrecord_dataset.prefetch(args.test_batch_size)
            self.test_iterator = self.test_tfrecord_dataset.make_initializable_iterator()
            self.iter_init_op = self.test_iterator.initializer
            self.test_input = self.test_iterator.get_next()

            train_tfrecord_dataset = TFData.TFRecordDataset(filenames=test_Files[1], compression_type='GZIP', num_parallel_reads=args.num_threads)
            train_tfrecord_dataset = train_tfrecord_dataset.map(self._parse_function, num_parallel_calls=args.num_threads)
            train_tfrecord_dataset = train_tfrecord_dataset.batch(args.test_batch_size)
            train_tfrecord_dataset = train_tfrecord_dataset.prefetch(args.test_batch_size)
            train_iterator = train_tfrecord_dataset.make_initializable_iterator()
            self.train_iter_init_op = train_iterator.initializer
            self.train_input = train_iterator.get_next()

    def read_tfrecord(self, data_path):
        if self.args.mode == 'train':
            TFR_path = os.path.join(data_path, '20200120_normal_tfrecord', '*.tfrecord')
            Files = TFData.Dataset.from_tensor_slices(glob(TFR_path))
            return Files, TFR_path

        elif self.args.mode == 'test':
            test_TFR_path = os.path.join(data_path[0], '20191208_data_testMK_ver5', '*.tfrecord')
            train_TFR_path = os.path.join(data_path[1], 'BTT_image_train2_tfrecord_20191122', '*.tfrecord')

            self.test_TFR_paths = sorted(glob(test_TFR_path))
            self.train_TFR_paths = sorted(glob(train_TFR_path))

            test_Files = TFData.Dataset.from_tensor_slices(self.test_TFR_paths)
            train_Files = TFData.Dataset.from_tensor_slices(self.train_TFR_paths)

            return [test_Files, train_Files], test_TFR_path

    def _parse_function(self, serialized_data):
        keys_to_features = {'Image': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(serialized_data, keys_to_features)

        if self.args.model_type == 'CVAE_v2':
            parsed_features['Image'] = tf.decode_raw(parsed_features['Image'], tf.uint8)
            reshape_data = tf.cast(tf.reshape(parsed_features['Image'], [256, 256, 3]), tf.float32)
            return reshape_data

        elif self.args.model_type == 'AE':
            parsed_features['Image'] = tf.decode_raw(parsed_features['Image'], tf.float32)
            reshape_data = tf.cast(tf.reshape(parsed_features['Image'], [600, 600, 1]), tf.float32)
            return reshape_data

        elif self.args.model_type == 'VAE':
            parsed_features['Image'] = tf.decode_raw(parsed_features['Image'], tf.float32)
            reshape_data = tf.cast(tf.reshape(parsed_features['Image'], [1, 1200, 1]), tf.float32)
            return reshape_data

    def train_preprocess(self, data):
        gamma = tf.random_uniform([], 0.1, 0.5)
        data = tf.cond(gamma > 0.3, lambda: data ** gamma, lambda: data)

        # Random gamma augmentation
        do_augment = tf.random_uniform([], 0, 1)
        data = tf.cond(do_augment > 0.5, lambda: self.augment_image(data), lambda: data)

        return data

    @staticmethod
    def augment_image(data):
        data_aug = tf.image.random_crop(data, size=[600, 600, 3])

        return data_aug