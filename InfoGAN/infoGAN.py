from __future__ import division
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics


from source import *
from other import *

class Discriminator(tf.keras.Model):
    def __init__(self, is_training=True):
        super(Discriminator, self).__init__(name='discriminator')
        self.is_training = is_training
        self.conv_1 = Conv2D(64, 4, 2)
        self.conv_2 = Conv2D(128, 4, 2)
        self.bn_1 = BatchNorm(is_training=self.is_training)
        self.bn_2 = BatchNorm(is_training=self.is_training)
        self.fc_1 = DenseLayer(1024)
        self.fc_2 = DenseLayer(1)

    def call(self, inputs, training):
        x = self.conv_1(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = self.conv_2(x)
        x = self.bn_1(x, training)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Flatten()(x)
        x = self.fc_1(x)
        x = self.bn_2(x, training)
        x = layers.LeakyReLU(alpha=0.2)(x)
        out_logits = self.fc_2(x)
        out = keras.activations.sigmoid(out_logits)

        return out, out_logits, x


class Generator(tf.keras.Model):
    def __init__(self, is_training=True):
        super(Generator, self).__init__(name='generator')
        self.is_training = is_training
        self.bn_1 = BatchNorm(is_training=self.is_training)
        self.bn_2 = BatchNorm(is_training=self.is_training)
        self.bn_3 = BatchNorm(is_training=self.is_training)
        self.fc_1 = DenseLayer(1024)
        self.fc_2 = DenseLayer(128*7*7)
        self.up_conv_1 = UpConv2D(64, 4, 2)
        self.up_conv_2 = UpConv2D(1, 4, 2)

    def call(self, inputs, training):
        x = self.fc_1(inputs)
        x = self.bn_1(x, training)
        x = layers.ReLU()(x)
        x = self.fc_2(x)
        x = self.bn_2(x, training)
        x = layers.ReLU()(x)
        x = layers.Reshape((7, 7, 128))(x)
        x = self.up_conv_1(x)
        x = self.bn_3(x, training)
        x = layers.ReLU()(x)
        x = self.up_conv_2(x)
        x = keras.activations.sigmoid(x)
        return x


class Classifier(tf.keras.Model):
    def __init__(self, y_dim, is_training=True):
        super(Classifier, self).__init__(name='classifier')
        self.is_training = is_training
        self.y_dim = y_dim
        self.bn_1 = BatchNorm(is_training=self.is_training)
        self.fc_1 = DenseLayer(64)
        self.fc_2 = DenseLayer(self.y_dim)

    def call(self, inputs, training):
        x = self.fc_1(inputs)
        x = self.bn_1(x, training)
        x = layers.LeakyReLU(alpha=0.2)(x)
        out_logits = self.fc_2(x)
        out = keras.layers.Softmax()(out_logits)
        return out, out_logits

class infoGAN():
    def __init__(self):
        super(infoGAN, self).__init__()
        self.model_name = 'infoGAN'
        self.batch_size =64
        self.z_dim = 62
        self.y_dim = 12
        self.len_discrete_code = 10  # categorical distribution (i.e. label)
        self.len_continuous_code = 2 # gaussian distribution (e.g. rotation, thickness)
        self.checkpoint_dir = os.path.join('checkpoint', self.model_name)
        self.datasets_name = 'mnist'
        self.log_dir ='logs'
        self.learnning_rate = 2e-4
        self.epoches = 20
        self.sample_z = tf.random.uniform(minval=-1, maxval=1, shape=(self.batch_size, self.z_dim),
                                          dtype=tf.dtypes.float32)
        self.datasets = load_mnist_data()
        self.g = Generator(is_training=True)
        self.d = Discriminator(is_training=True)
        self.c = Classifier(y_dim=self.y_dim, is_training=True)
        self.g_optimizer = keras.optimizers.Adam(lr=5*self.learnning_rate, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(lr=self.learnning_rate, beta_1=0.5)
        self.q_optimizer = keras.optimizers.Adam(lr=5*self.learnning_rate, beta_1=0.5)
        self.g_loss_metric = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss_metric = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        self.q_loss_metric = tf.keras.metrics.Mean('q_loss', dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=self.g_optimizer,
                                              discriminator_optimizer=self.d_optimizer,
                                              classifier_optimizer=self.q_optimizer,
                                              generator=self.g,
                                              discriminator=self.d,
                                              classifier=self.c)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    def d_loss_fun(self, d_fake_logits, d_real_logits):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits))
        total_loss = d_loss_fake+d_loss_real
        return total_loss

    def g_loss_fun(self, logits):
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits))
        return g_loss

    def q_loss_fun(self, disc_code_est, disc_code_tg, cont_code_est, cont_code_tg):
        q_disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=disc_code_tg, logits=disc_code_est))
        q_cont_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(cont_code_tg - cont_code_est), axis=1))
        q_loss = q_disc_loss+q_cont_loss
        return q_loss


    # training for one batch

    def train_one_step(self, batch_labels, batch_images):
        noises = tf.random.uniform(shape=(self.batch_size, self.z_dim), minval=-1, maxval=1, dtype=tf.dtypes.float32)
        code = tf.random.uniform(minval=-1, maxval=1, shape=(self.batch_size, self.len_continuous_code), dtype=tf.dtypes.float32)
        batch_codes = tf.concat((batch_labels, code), axis=1)
        batch_z = tf.concat([noises, batch_codes], 1)
        real_images = conv_cond_concat(batch_images, batch_codes)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as q_tape:
            fake_imgs = self.g(batch_z, training=True)
            fake_imgs = conv_cond_concat(fake_imgs, batch_codes)
            d_fake, d_fake_logits, input4classifier_fake = self.d(fake_imgs, training=True)
            d_real, d_real_logits, _ = self.d(real_images, training=True)
            d_loss = self.d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = self.g_loss_fun(d_fake_logits)
            code_fake, code_logit_fake = self.c(input4classifier_fake, training=True)
            disc_code_est = code_logit_fake[:, :self.len_discrete_code]
            disc_code_tg = batch_codes[:, :self.len_discrete_code]
            cont_code_est = code_logit_fake[:, self.len_discrete_code:]
            cont_code_tg = batch_codes[:, self.len_discrete_code:]
            q_loss = self.q_loss_fun(disc_code_est, disc_code_tg, cont_code_est, cont_code_tg)

        gradients_of_d = d_tape.gradient(d_loss, self.d.trainable_variables)
        gradients_of_g = g_tape.gradient(g_loss, self.g.trainable_variables)
        gradients_of_q = q_tape.gradient(q_loss, self.c.trainable_variables)

        self.d_optimizer.apply_gradients(zip(gradients_of_d, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_g, self.g.trainable_variables))
        self.q_optimizer.apply_gradients(zip(gradients_of_q, self.c.trainable_variables))

        self.g_loss_metric(g_loss)
        self.d_loss_metric(d_loss)
        self.q_loss_metric(q_loss)

    def train(self):

        self.sample_label=tf.cast(2*(tf.ones(shape=(self.batch_size,),dtype=tf.int32)),dtype=tf.int32)
        self.sample_label=tf.one_hot(self.sample_label,depth=10)
        for epoch in range(self.epoches):#自己定一个self.epoches
            for batch_images, batch_labels in self.datasets:
                self.continuous_code=tf.random.uniform(shape=[self.batch_size, self.len_continuous_code],
                                                        minval=-1.0,maxval=1.0,dtype=tf.dtypes.float32)
                self.test_codes = tf.concat([self.sample_label,self.continuous_code],1)
                self.train_one_step(batch_labels, batch_images)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)


                #save checkpoints for every 400 batches training
                if step % 100 == 0:
                    print('step： {}, d_loss: {:.4f}, g_loss: {:.4F}, q_loss: {:.4F}'
                    .format(step, self.d_loss_metric.result(), self.g_loss_metric.result(),self.q_loss_metric.result()))
                    manifold_h = int(np.floor(np.sqrt(self.batch_size)))
                    manifold_w = int(np.floor(np.sqrt(self.batch_size)))
                    self.batch_z_to_disply = tf.concat([self.sample_z, self.test_codes[:self.batch_size, :]], 1)
                    result_to_display = self.g(self.batch_z_to_disply, training=False)
                    save_images(result_to_display[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + check_folder('results' ) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, int(step)))

def main():
    model = infoGAN()
    model.train()

if __name__ == '__main__':
    main()