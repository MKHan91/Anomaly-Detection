import tensorflow as tf
import tensorflow.contrib.slim as slim


class VAE_models(object):
    def __init__(self, model_input, args, reuse_variables=None):
        super(VAE_models, self).__init__()

        self.model_input = model_input
        self.args = args
        self.reuse_variables = reuse_variables
        self.model_collection = ['model_0']

        self.build_model()

        if args.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)

        mu_x = slim.avg_pool2d(x, kernel_size=3, stride=1, padding='SAME')
        mu_y = slim.avg_pool2d(y, kernel_size=3, stride=1, padding='SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value(1 - SSIM, 0, 1)

    def denseconv(self, x, num_outputs, kernel_size, stride=1, dilation_rate=1):
        out = slim.batch_norm(x)
        out = tf.nn.relu(out)
        out = slim.conv2d(out, num_outputs, kernel_size, stride, rate=dilation_rate)
        return out

    def max_denseblock(self, x, num_outputs, kernel_size, stride=1, dilation_rate=1):
        out = slim.max_pool2d(x, kernel_size, stride=2, padding='SAME')
        conv_out = slim.conv2d(out, num_outputs, kernel_size, stride, rate=dilation_rate)

        out = self.denseconv(conv_out, num_outputs*4, kernel_size)
        out = self.denseconv(out, num_outputs, kernel_size)
        out = tf.concat([conv_out, out], axis=3)
        return out

    def conv_denseblock(self, x, num_outputs, kernel_size, stride=1, dilation_rate=1):
        conv_out = slim.conv2d(x, num_outputs, kernel_size, 2, rate=dilation_rate)

        out = self.denseconv(conv_out, num_outputs*4, kernel_size, stride, dilation_rate=dilation_rate)
        out = self.denseconv(out, num_outputs, kernel_size, stride, dilation_rate=dilation_rate)
        out = tf.concat([conv_out, out], axis=3)
        return out

    def build_VAE(self):
        with tf.variable_scope('Encoder', reuse=self.reuse_variables):
            """
            Q(Z|X)(approximate posterior distribution) encoder : Multivariate Gaussian
            """
            self.en_layer1 = slim.fully_connected(inputs=self.model_input, num_outputs=self.args.h_dim, scope='en_fc1')
            self.en_layer2 = slim.fully_connected(inputs=self.en_layer1, num_outputs=self.args.h_dim, scope='en_fc2',
                                                  normalizer_fn=slim.batch_norm)
            en_layer3 = slim.fully_connected(inputs=self.en_layer2, num_outputs=int(self.args.h_dim/2), scope='en_fc3',
                                             normalizer_fn=slim.batch_norm)
            en_layer4 = slim.fully_connected(inputs=en_layer3, num_outputs=int(self.args.h_dim/2), scope='en_fc4',
                                             normalizer_fn=slim.batch_norm)
            en_layer5 = slim.fully_connected(inputs=en_layer4, num_outputs=int(self.args.h_dim / 4), scope='en_fc5',
                                             normalizer_fn=slim.batch_norm)
            en_layer6 = slim.fully_connected(inputs=en_layer5, num_outputs=int(self.args.h_dim / 4), scope='en_fc6',
                                             normalizer_fn=slim.batch_norm)

            self.z_mu = slim.fully_connected(inputs=en_layer6, num_outputs=self.args.z_dim, activation_fn=None, scope='mu')
            self.z_log_sigma = slim.fully_connected(inputs=en_layer5, num_outputs=self.args.z_dim, activation_fn=None, scope='log_sigma')

        with tf.variable_scope('Sampling_z'):
            """
            sampling z using reparameterization trick
            """
            epsilon = tf.random_normal(shape=tf.shape(self.z_mu), dtype=tf.float32)
            self.z_sample = self.z_mu + tf.exp(self.z_log_sigma / 2.) * epsilon

        with tf.variable_scope('Decoder', reuse=self.reuse_variables):
            """
            P(X|Z) (likelihood)
            """
            self.de_layer1 = slim.fully_connected(inputs=en_layer6, num_outputs=int(self.args.h_dim/4), scope='de_fc1',
                                                  normalizer_fn=slim.batch_norm)
            self.de_layer2 = slim.fully_connected(inputs=self.de_layer1, num_outputs=int(self.args.h_dim/4), scope='de_fc2',
                                                  normalizer_fn=slim.batch_norm)
            de_layer3 = slim.fully_connected(inputs=self.de_layer2, num_outputs=int(self.args.h_dim/2), scope='de_fc3',
                                             normalizer_fn=slim.batch_norm)
            de_layer4 = slim.fully_connected(inputs=de_layer3, num_outputs=int(self.args.h_dim/2), scope='de_fc4',
                                             normalizer_fn=slim.batch_norm)
            de_layer5 = slim.fully_connected(inputs=de_layer4, num_outputs=self.args.h_dim, scope='de_fc5',
                                             normalizer_fn=slim.batch_norm)
            self.logits = slim.fully_connected(inputs=de_layer5,
                                               num_outputs=1200,
                                               activation_fn=tf.nn.relu,
                                               scope='logits')
            self.probability = tf.nn.sigmoid(self.logits)

    def build_AE(self):
        with tf.variable_scope('Encoder', reuse=self.reuse_variables):
            "Initial conv block, H/2"
            net = slim.conv2d(inputs=self.model_input, num_outputs=32, kernel_size=7, stride=2, scope='conv1')
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

            "H/4"
            net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=3, stride=2)

            "H/8"
            net = slim.max_pool2d(net, kernel_size=3, stride=2, padding='SAME')
            conv2_net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=1, scope='conv2')

            "denseblock series, H/8"
            conv_dense = self.conv_denseblock(conv2_net, num_outputs=128, kernel_size=3)
            max_dense = self.max_denseblock(conv2_net, num_outputs=128, kernel_size=3)
            dense = tf.concat([conv_dense, max_dense], axis=3)

            "dilated dense convolution"
            net = self.denseconv(dense, num_outputs=512, kernel_size=3, dilation_rate=3)
            net = self.denseconv(net, num_outputs=512, kernel_size=3, dilation_rate=6)

            # "H/16"
            # net = slim.avg_pool2d(net, kernel_size=3, padding='SAME')
        with tf.variable_scope('Decoder', reuse=self.reuse_variables):
            "H/4"
            self.de_net = slim.conv2d_transpose(inputs=net, num_outputs=256, kernel_size=3, stride=2, scope='deconv1') # (B, 76, 76, 256)
            conv_denet = slim.conv2d(inputs=self.de_net, num_outputs=128, kernel_size=3)

            "H/2"
            self.de_net2 = slim.conv2d_transpose(inputs=conv_denet, num_outputs=64, kernel_size=3, stride=2,scope='deconv2') # (B, 152, 152, 128)
            conv_denet = slim.conv2d(inputs=self.de_net2, num_outputs=32, kernel_size=3)

            "H"
            self.de_net3 = slim.conv2d_transpose(inputs=conv_denet, num_outputs=16, kernel_size=3, stride=2, scope='deconv3')
            self.logits = slim.conv2d(inputs=self.de_net3, num_outputs=1, kernel_size=3)
            # (B, 304, 304, 64)

            # self.de_net4 = slim.conv2d_transpose(inputs=conv_denet, num_outputs=1, kernel_size=3, stride=2, scope='deconv4')
            # (B, 608, 608, 3)
            #
            # self.probability = tf.nn.sigmoid(self.logits)

    def build_CVAE_v2(self):
        with tf.variable_scope('Encoder', reuse=self.reuse_variables):
            """
            Q(Z|X)(approximate posterior distribution) encoder : Multivariate Gaussian
            """
            "Initial conv block, H/2"
            net = slim.conv2d(inputs=self.model_input, num_outputs=32, kernel_size=7, stride=2, scope='conv1')
            net = slim.batch_norm(net, is_training=False)
            net = tf.nn.relu(net)
            net = slim.max_pool2d(net, 3, stride=2, padding='SAME')
            
            # "H/4"
            # net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=2)
            # net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=1)
            #
            # "H/8"
            # net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=3, stride=2)
            # net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=3, stride=1)
            #
            # "H/16"
            # net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=3, stride=2)
            # net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=3, stride=1)
            #
            # "H/32"
            # net = slim.conv2d(inputs=net, num_outputs=512, kernel_size=3, stride=2)
            # net = slim.conv2d(inputs=net, num_outputs=512, kernel_size=3, stride=1)

            # "H/4"
            # net = slim.max_pool2d(net, kernel_size=3, stride=2, padding='SAME')
            #
            # "H/8"
            # net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=2)
            #
            # "dilated dense convolution"
            # net = self.denseconv(net, num_outputs=64, kernel_size=3, dilation_rate=3)
            # net = self.denseconv(net, num_outputs=64, kernel_size=3, dilation_rate=6)
            #
            # "denseblock series, H/8"
            # conv_dense = self.conv_denseblock(net, num_outputs=128, kernel_size=3)
            # max_dense = self.max_denseblock(net, num_outputs=128, kernel_size=3)
            # dense = tf.concat([conv_dense, max_dense], axis=3)
            #
            # "H/32"
            # net = self.conv_denseblock(dense, num_outputs=512, kernel_size=3)

            "mean / variation"
            self.z_mu = slim.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME')
            self.z_log_var = slim.avg_pool2d(net ** 2, 3, 1, 'SAME') - self.z_mu ** 2

            with tf.variable_scope('Sampling_z'):
                """
                sampling z using reparameterization trick
                """
                epsilon = tf.random_normal(shape=tf.shape(self.z_mu), dtype=tf.float32)
                self.z_sample = self.z_mu + tf.exp(self.z_log_var / 2.) * epsilon

        with tf.variable_scope('Decoder', reuse=self.reuse_variables):
            """
            P(X|Z) (likelihood)
            """
            self.de_net = slim.conv2d_transpose(inputs=self.z_sample, num_outputs=256, kernel_size=3, stride=2, scope='deconv1')
            self.de_net = slim.conv2d_transpose(inputs=self.de_net, num_outputs=128, kernel_size=3, stride=2, scope='deconv2')
            self.de_net = slim.conv2d_transpose(inputs=self.de_net, num_outputs=32, kernel_size=3, stride=2, scope='deconv3')
            self.de_net = slim.conv2d_transpose(inputs=self.de_net, num_outputs=16, kernel_size=3, stride=2, scope='deconv4')
            # conv = slim.conv2d(self.de_net, num_outputs=8, kernel_size=3, stride=1)
            self.logits = slim.conv2d_transpose(inputs=self.de_net, num_outputs=3, kernel_size=3, stride=2,
                                                activation_fn=tf.nn.elu)

            # self.logits = slim.conv2d(inputs=self.de_net, num_outputs=3, kernel_size=1)

    def build_model(self):
        with tf.variable_scope('mk_model', reuse=self.reuse_variables):
            if self.args.model_type == 'VAE':
                self.build_VAE()

            elif self.args.model_type == 'AE':
                self.build_AE()

            elif self.args.model_type == 'CVAE_v2':
                self.build_CVAE_v2()

    def build_losses(self):
        with tf.variable_scope('Losses', reuse=self.reuse_variables):
            """
            loss function = - KL divergence loss + reconstruction loss
            """
            if self.args.model_type == 'AE':
                # Reconstruction loss
                self.reconstruction_loss = tf.reduce_mean(
                    tf.keras.backend.binary_crossentropy(target=self.model_input, output=self.logits))

                # L1 reconstruction loss
                self.l1_recon_loss = tf.reduce_mean(tf.abs(self.model_input - self.logits))

                # SSIM
                self.ssim = [self.SSIM(self.logits[i, :, :, :], self.model_input[i, :, :, :]) for i in range(self.args.batch_size)]
                self.mean_ssim = tf.reduce_mean(self.ssim)

                self.ssim_recon_loss = self.mean_ssim * self.reconstruction_loss

                self.total_loss = (1 - self.args.weight_ratio) * self.ssim_recon_loss + self.args.weight_ratio * self.l1_recon_loss

            elif self.args.model_type == 'CVAE_v2':
                # KL Divergence D_KL( Q(Z|X) || P(Z|X) )
                D_KL = 1. + self.z_log_var - tf.exp(self.z_log_var) - self.z_mu**2
                self.KL_Loss = -0.5 * tf.reduce_sum(D_KL)
                # self.KL_loss = -0.5 * tf.reduce_sum(tf.reshape(D_KL, [-1, D_KL.shape[1]*D_KL.shape[2]*D_KL.shape[3]]), 1)

                # Reconstruction loss
                # self.loss_recon = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=self.model_input, y_pred=self.logits))
                # self.loss_recon = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.model_input, logits=self.logits))
                self.loss_recon = tf.reduce_mean(tf.abs(self.model_input - self.logits))
                self.total_loss = self.KL_Loss + self.loss_recon

    def build_summaries(self):
        with tf.device('/cpu:0'):
            if self.args.model_type == 'AE':
                tf.summary.scalar(name='reconstruction_loss', tensor=self.reconstruction_loss, collections=self.model_collection)
                tf.summary.scalar(name='l1_recon_loss', tensor=tf.reduce_mean(self.l1_recon_loss), collections=self.model_collection)
                tf.summary.scalar(name='ssim', tensor=self.mean_ssim, collections=self.model_collection)
                tf.summary.scalar(name='ssim_recon_loss', tensor=self.ssim_recon_loss, collections=self.model_collection)

            elif self.args.model_type == 'CVAE_v2':
                tf.summary.image('Input_image', tensor=self.model_input, max_outputs=4, collections=self.model_collection)
                tf.summary.image('Estimation_image', tensor=self.logits, max_outputs=4, collections=self.model_collection)
                tf.summary.scalar(name='KL_loss', tensor=self.KL_Loss, collections=self.model_collection)
                tf.summary.scalar(name='reconstruction_loss', tensor=self.loss_recon, collections=self.model_collection)