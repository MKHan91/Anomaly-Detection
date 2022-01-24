from autoencoder_dataloader import AutoencoderDataloader
from arch import VAE_models
from average_gradients import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser(description='autoencoder of Myungkyu')
parser.add_argument('--mode',               type=str,   help='train or test',                                           default='train')
parser.add_argument('--model_name',         type=str,   help='model_name',                                              default='')
parser.add_argument('--model_type',         type=str,   help='type of model',                                           default='')
parser.add_argument('--train_data_path',    type=str,   help='path to the train data',                                  default='')
parser.add_argument('--test_data_path',     type=str,   help='path to the test data',                                   default='')
parser.add_argument('--batch_size',         type=int,   help='batch size of train data',                                default=4)
parser.add_argument('--test_batch_size',    type=int,   help='batch size of test data',                                 default=3)
parser.add_argument('--num_epochs',         type=int,   help='number of epochs',                                        default=300)
parser.add_argument('--learning_rate',      type=float, help='initial learning rate',                                   default=1e-4)
parser.add_argument('--weight_ratio',       type=float, help='l1 recon loss weight ratio',                              default=0.85)
parser.add_argument('--end_learning_rate',  type=float, help='end learning rate',                                       default=-1)
parser.add_argument('--num_gpus',           type=int,   help='the number of GPUs for training',                         default=1)
parser.add_argument('--num_threads',        type=int,   help='number of threads for data loading',                      default=12)
parser.add_argument('--log_directory',      type=str,   help='path to a specific checkpoint to load',                   default='')
parser.add_argument('--checkpoint_path',    type=str,   help='path to a specific checkpoint to load',                   default='')
parser.add_argument('--output_directory',   type=str,   help='output directory for test',                               default='')
parser.add_argument('--retrain',          help='if used with checkpoint_path, will restart training from step zero',    action='store_true')
parser.add_argument('--fully_summary',      type=bool, help='if set, will keep more data for each summary.',            default=True)

args = parser.parse_args()

def make_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def train(calendar):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_steps = tf.Variable(0, name='global_step', trainable=False)

        dataloader = AutoencoderDataloader(args=args)
        model_input = dataloader.model_input
        iter_init_op = dataloader.iter_init_op

        # OPTIMIZER
        training_num_samples = dataloader.training_num_samples
        steps_per_epoch = np.ceil(training_num_samples / args.batch_size).astype(np.int32)
        num_total_steps = args.num_epochs * steps_per_epoch

        start_learning_rate = args.learning_rate
        end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else start_learning_rate * 0.1
        learning_rate = tf.train.polynomial_decay(learning_rate=start_learning_rate,
                                                  global_step=global_steps,
                                                  decay_steps=num_total_steps,
                                                  end_learning_rate=end_learning_rate,
                                                  power=0.9)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        reuse_variables = None
        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    model = VAE_models(model_input=model_input, args=args)

                    loss = model.total_loss

                    reuse_variables = True
                    gradients = opt_step.compute_gradients(loss)
                    tower_grads.append(gradients)

        gradients = average_gradients(tower_grads=tower_grads)
        apply_gradients = opt_step.apply_gradients(grads_and_vars=gradients, global_step=global_steps)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # SAVER
        date = '_'.join(calendar)
        thedate = '_'.join(calendar[0:3])
        save_path = os.path.join(args.log_directory, args.model_name + '_' + thedate)
        log_path = os.path.join(save_path, args.model_name + '_' + date)
        check_dir_or_create(dir=save_path)
        check_dir_or_create(dir=log_path)

        summary_writer = tf.summary.FileWriter(logdir=log_path)
        train_saver = tf.train.Saver(max_to_keep=10)

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("The number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # THREAD
        # coordinator = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT
        if args.checkpoint_path != '':
            train_saver.restore(sess=sess, save_path=args.checkpoint_path)

            if args.retrain:
                sess.run(global_steps.assign(args.checkpoint_path.split('0')))

        # RUN!
        start_step = global_steps.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            if step % steps_per_epoch == 0:
                sess.run(iter_init_op)
            _, lr_value, loss_value = sess.run([apply_gradients, learning_rate, loss])
            # while True:
            #     try:
            #         _, lr_value, loss_value, entropy_loss = sess.run([apply_gradients, learning_rate, loss, cross_entropy])
            #     except tf.errors.OutOfRangeError:
            #         break
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = args.batch_size / duration
                time_sofar = (time.time() - start_time)
                print_string = 'step: {}/{} | total loss: {:.5f} |learning_rate: {:.12f} | time elapsed: {:.1f}s | examples per sec: {:.1f}'
                print(print_string.format(step, num_total_steps, loss_value, lr_value, time_sofar, examples_per_sec))

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

                # SAVE THE MODEL
                train_saver.save(sess=sess, save_path=log_path + '/model', global_step=step)

        # SAVE THE LASTEST MODEL
        train_saver.save(sess=sess, save_path=log_path + '/model', global_step=num_total_steps)
        print("Training Done!")

def test():
    dataloader = AutoencoderDataloader(args=args)

    # test_num_samples = dataloader.test_num_samples
    # test_input = dataloader.test_input
    # train_input = dataloader.train_input
    #
    # model_input = tf.placeholder(dtype=tf.float32, shape=(None, 600, 600, 1))
    # model = VAE_models(model_input=model_input, args=args)
    #
    # SAVE_PATH = os.path.join(args.output_directory, args.model_name+'_ver6.2')
    #
    # # SESSION
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    #
    # # INIT
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    #
    # print('now testing {} files'.format(test_num_samples))
    # sess.run(dataloader.iter_init_op)
    # sess.run(dataloader.train_iter_init_op)
    #
    # # SAVER
    # train_saver = tf.train.Saver()
    #
    # with tf.device('/cpu:0'):
    #     restore_path = os.path.join(args.log_directory, 'BTT_AE_2019_11_26', args.model_name, 'model-4450')
    #     # RESTORE
    #     train_saver.restore(sess=sess, save_path=restore_path)
    #
    #     # PREDICTION
    #     sess_test_input = sess.run(test_input)
    #     sess_train_input = sess.run(train_input)
    #
    #     test_prediction = sess.run(model.logits, feed_dict={model_input:sess_test_input})
    #     train_prediction = sess.run(model.logits, feed_dict={model_input:sess_train_input})
    #     # prediction_prob = sess.run(model.probability)
    #     # print(prediction_prob)
    #
    #     print('Calculating RMSE...')
    #     # RMSE_list = []
    #     # train_RMSE_list = []
    #     # for file_num in range(args.test_batch_size):
    #     # List initialization
    #     RMSE_list = []
    #     train_RMSE_list = []
    #     for bld_num in range(1, 61):
    #         # MAKE DIRECTORY
    #         make_directory(dir=SAVE_PATH)
    #
    #         RMSE = tf.sqrt(
    #             tf.reduce_mean(
    #                 tf.square(test_prediction[0, (10*bld_num-10):10*bld_num, :, :] -
    #                           sess_test_input[0, (10*bld_num-10):10*bld_num, :, :])
    #             )
    #         )
    #
    #         train_RMSE = tf.sqrt(
    #             tf.reduce_mean(
    #                 tf.square(train_prediction[0, (10*bld_num-10):10*bld_num, :, :] -
    #                           sess_train_input[0, (10*bld_num-10):10*bld_num, :, :])
    #             )
    #         )
    #
    #         RMSE_list += [RMSE]
    #         train_RMSE_list += [train_RMSE]
    #
    #         print('current processing: {}/{}, file_number: {}/{}'.format(bld_num, 60, 0+1, test_num_samples))
    #
    #         # PLOTTING
    #         plt.title('Anomaly score of {} BLADE 05_2bent '.format(bld_num))
    #         plt.xlabel('Blade number')
    #         plt.ylabel('Anomaly score')
    #         plt.xticks(np.arange(1, 60, 9.0))
    #         plt.plot(sess.run(RMSE_list), '*', color='red', label='test')
    #         plt.plot(sess.run(train_RMSE_list), '.', color='blue', label='train')
    #         # plt.xlim([-1, 10])
    #         # plt.ylim([min(sess.run(train_RMSE_list))-0.005, max(sess.run(RMSE_list))+0.01])
    #         plt.ylim([0.0020, 0.2])
    #         plt.legend(loc='best')
    #     plt.savefig(SAVE_PATH+'/Anomaly score of {:02d} BLADE - bent 05_2bent.png'.format(bld_num))
    #     plt.show()

    test_num_samples = dataloader.test_num_samples
    test_input = dataloader.test_input
    train_input = dataloader.train_input

    model_input = tf.placeholder(dtype=tf.float32, shape=(None, 600, 600, 1))
    model = VAE_models(model_input=model_input, args=args)

    SAVE_PATH = os.path.join(args.output_directory, args.model_name+'_ver1.0')

    # SESSION
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('now testing {} files'.format(test_num_samples))
    sess.run(dataloader.iter_init_op)
    sess.run(dataloader.train_iter_init_op)

    # SAVER
    train_saver = tf.train.Saver()

    with tf.device('/cpu:0'):
        restore_path = os.path.join(args.log_directory, 'BTT_AE_2019_12_07', args.model_name, 'model-133500')
        # RESTORE
        train_saver.restore(sess=sess, save_path=restore_path)

        # PREDICTION
        sess_test_input = sess.run(test_input)
        sess_train_input = sess.run(train_input)

        test_prediction = sess.run(model.logits, feed_dict={model_input:sess_test_input})
        train_prediction = sess.run(model.logits, feed_dict={model_input:sess_train_input})
        # prediction_prob = sess.run(model.probability)
        # print(prediction_prob)

        print('Calculating RMSE...')
        # List initialization
        RMSE_list = []
        train_RMSE_list = []
        for num_batch in range(args.test_batch_size):
            for bld_num in range(1, 61):
                # MAKE DIRECTORY
                make_directory(dir=SAVE_PATH)

                RMSE = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(test_prediction[num_batch, (10*bld_num-10):10*bld_num, :, :] -
                                  sess_test_input[num_batch, (10*bld_num-10):10*bld_num, :, :])
                    )
                )

                train_RMSE = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(train_prediction[num_batch, (10*bld_num-10):10*bld_num, :, :] -
                                  sess_train_input[num_batch, (10*bld_num-10):10*bld_num, :, :])
                    )
                )

                RMSE_list += [RMSE]
                train_RMSE_list += [train_RMSE]

                print('current processing: {}/{}, file_number: {}/{}'.format(bld_num, 60, num_batch + 1, test_num_samples))

            # PLOTTING
            plt.title('Anomaly score of {} BLADE '.format(bld_num))
            plt.xlabel('Blade number')
            plt.ylabel('Loss_RMSE')
            plt.xticks(np.arange(1, 60, 9.0))
            # ax1.plot(sess.run(RMSE_list), '*', color='red', label='test')
            # ax1.plot(sess.run(train_RMSE_list), '.', color='blue', label='train')
            if num_batch == 0:
                plt.plot(np.arange(1, 61), sess.run(train_RMSE_list)[0:60], '.', color='blue', label='train')
                plt.plot(np.arange(1, 61), sess.run(RMSE_list)[0:60], '.', color='red', label='test norm')
                # plt.ylim([0.002, 0.05])
                # plt.legend(loc='best')
                # plt.show()
            elif num_batch == 1:
                plt.plot(np.arange(1, 61), sess.run(RMSE_list)[60:120], '*', color='green', label='10Hz')
                # plt.plot(np.arange(1, 61), sess.run(train_RMSE_list)[0:60], '.', color='blue', label='train')
                # plt.ylim([0.002, 0.05])
                # plt.legend(loc='best')
                # plt.show()
            elif num_batch == 2:
                plt.plot(np.arange(1, 61), sess.run(RMSE_list)[120:180], '*', color='purple', label='20Hz')
                # plt.plot(np.arange(1, 61), sess.run(train_RMSE_list)[0:60], '.', color='blue', label='train')
                # plt.ylim([0.002, 0.05])
                # plt.legend(loc='best')
                # plt.show()
            elif num_batch == 3:
                plt.plot(np.arange(1, 61), sess.run(RMSE_list)[180:240], '*', color='black', label='0.60 freq')
                # plt.plot(sess.run(train_RMSE_list)[0:60], '.', color='blue', label='train')
                # plt.ylim([0.002, 0.05])
                # plt.legend(loc='best')
                # plt.show()
            elif num_batch == 4:
                plt.plot(np.arange(1, 61), sess.run(RMSE_list)[240:300], '*', color='darkorange', label='0.80 freq')
                plt.plot(sess.run(train_RMSE_list)[0:60], '.', color='blue', label='train')
                plt.ylim([0.002, 0.05])
                plt.legend(loc='best')
                plt.show()
            elif num_batch == 5:
                plt.plot(np.arange(1, 61), sess.run(RMSE_list)[300:360], '*', color='magenta', label='1.00 freq')
                plt.plot(sess.run(train_RMSE_list)[0:60], '.', color='blue', label='train')
                plt.ylim([0.002, 0.09])
                plt.legend(loc='best')
                plt.show()
        plt.ylim([0.002, 0.05])
        plt.legend(loc='best')
    plt.show()
    print('Test done!!')


def main():
    command = "mkdir " + os.path.join(os.getcwd(), "arch.py")
    os.system(command)

    command = "mkdir " + os.path.join(os.getcwd(), "autoencoder_dataloader.py")
    os.system(command)
    
    command = "mkdir " + os.path.join(os.getcwd(), "autoencoder_main.py")
    os.system(command)

    if args.mode == 'train':
        train()

    elif args.mode == 'test':
        test()

if __name__ == '__main__':
    main()