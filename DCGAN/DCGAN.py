# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:25:43 2020

@author: XiaoBanni
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,optimizers,losses,datasets
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_moons 
from sklearn.model_selection import train_test_split
from PIL import Image
import imageio
import  glob

#%%
'''
加载数据集
'''
from dataset import make_anime_dataset

#读取数据集路径，从https://pan.baidu.com/s/1eSifHcA 提取码：g5qa 下载解压
img_path=glob.glob(r'faces/*.jpg')
#glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件
#glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）；
#该方法需要一个参数用来指定匹配的路径字符串（字符串可以为绝对路径也可以为相对路径），其返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件。
batch_size=128
dataset,img_shape,_=make_anime_dataset(img_path, batch_size=batch_size,resize=64)
#其中dataset对象就是tf.data.Dataset类实例，已经完成了随机打散、预处理和批量化等操作，可以直接迭代获得样本批，img_shape是预处理后的图片大小。
dataset=dataset.repeat(300)

#%%
'''
生成器

生成网络G由 5 个转置卷积层单元堆叠而成，实现特征图高宽的层层放大，特征图通道数的层层减少。
首先将长度为100的隐藏向量𝒛通过Reshape操作调整为[𝑏, 1,1,100]的4维张量，
并依序通过转置卷积层，放大高宽维度，减少通道数维度，最后得到高宽为 64，通道数为3的彩色图片。
每个卷积层中间插入BN层来提高训练稳定性，卷积层选择不使用偏置向量。
'''

class Generator(keras.Model):
    #生成器网络类
    def __init__(self):
        super(Generator,self).__init__()
        filter=64
        #转置卷积层1，输出channel为filter*8,核大小4，步长1,不使用padding，不使用偏置
        #当设置 padding=’VALID’时，输出大小表达为：o=(i-1)s+k
        self.conv1=layers.Conv2DTranspose(filter*8,4,1,'valid',use_bias=False)
        self.bn1=layers.BatchNormalization()
        #转置卷积层2
        self.conv2=layers.Conv2DTranspose(filter*4,4,2,'same',use_bias=False)
        self.bn2=layers.BatchNormalization()
        #转置卷积层3
        self.conv3=layers.Conv2DTranspose(filter*2,4,2,'same',use_bias=False)
        self.bn3=layers.BatchNormalization()
        #转置卷积层4
        self.conv4=layers.Conv2DTranspose(filter*1,4,2,'same',use_bias=False)
        self.bn4=layers.BatchNormalization()
        #转置卷积5
        self.conv5=layers.Conv2DTranspose(3,4,2,'same',use_bias=False)
        
    def call(self,inputs,training=None):
        x=inputs #[z,100]
        #Reshape为[b,1,1,100]
        x=tf.reshape(x,(x.shape[0],1,1,x.shape[1]))
        x=tf.nn.relu(x)
        #转置卷积-BN-激活函数[b,4,4,512]
        x=tf.nn.relu(self.bn1(self.conv1(x),training=training))
        #转置卷积-BN-激活函数[b,8,8,256]
        x=tf.nn.relu(self.bn2(self.conv2(x),training=training))
        #转置卷积-BN-激活函数[b,16,16,128]
        x=tf.nn.relu(self.bn3(self.conv3(x),training=training))
        #转置卷积-BN-激活函数[b,32,32,64]
        x=tf.nn.relu(self.bn4(self.conv4(x),training=training))
        #转置卷积-激活函数[b,64,64,3]
        x=self.conv5(x)
        x=tf.tanh(x)#输出x范围为-1~1，与预处理一致
        
        return x
    
#%%
'''
判别器

判别网络D与普通的分类网络相同，
接受大小为[𝑏,64,64,3]的图片张量，
连续通过5个卷积层实现特征的层层提取，
卷积层最终输出大小为[𝑏, 2,2,1024]，
再通过池化层 GlobalAveragePooling2D将特征大小转换为[𝑏, 1024]，
最后通过一个全连接层获得二分类任 务的概率。判别网络D类的代码实现如下：
'''

class Discriminator(keras.Model):
    #判别器类
    def __init__(self):
        super(Discriminator,self).__init__()
        filter=64
        #卷积层1
        self.conv1=layers.Conv2D(filter,4,2,'valid',use_bias=False)
        self.bn1=layers.BatchNormalization()
        #卷积层2
        self.conv2=layers.Conv2D(filter*2,4,2,'valid',use_bias=False)
        self.bn2=layers.BatchNormalization()
        #卷积层3
        self.conv3=layers.Conv2D(filter*4,4,2,'valid',use_bias=False)
        self.bn3=layers.BatchNormalization()
        #卷积层4
        self.conv4=layers.Conv2D(filter*8,3,1,'valid',use_bias=False)
        self.bn4=layers.BatchNormalization()
        #卷积层5
        self.conv5=layers.Conv2D(filter*16,3,1,'valid',use_bias=False)
        self.bn5=layers.BatchNormalization()
        #全局池化层
        self.pool=layers.GlobalAveragePooling2D()
        #特征打平层
        self.flatten=layers.Flatten()
        #二分类全连接层
        self.fc=layers.Dense(1)
        
    def call(self,inputs,training=None):
        #卷积-BN-激活函数:(4,31,31,64)
        # 卷积-BN-激活函数:(4, 31, 31, 64)
        
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # 卷积-BN-激活函数:(4, 4, 4, 512)
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        # 卷积-BN-激活函数:(4, 2, 2, 1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        # 卷积-BN-激活函数:(4, 1024)
        x = self.pool(x)
        # 打平
        x = self.flatten(x)
        # 输出，[b, 1024] => [b, 1]
        logits = self.fc(x)

        return logits      
    
        #判别器的输出大小为[𝑏, 1]，类内部没有使用 Sigmoid 激活函数，
        #通过 Sigmoid 激活函数后 可获得𝑏个样本属于真实样本的概率
    
#%%

'''
训练与可视化

'''

#%%

'''
判别网络

判别网络的训练目标是最大化ℒ(𝐷, 𝐺)函数，
使得真实样本预测为真的概率接近于 1，生成样本预测为真的概率接近于 0。
我们将判断器的误差函数实现 在 d_loss_fn 函数中，
将所有真实样本标注为 1，所有生成样本标注为 0，
并通过最小化对应的交叉熵损失函数来实现最大化ℒ(𝐷, 𝐺)函数。
'''

def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(generator,discriminator,batch_z,batch_x,is_training):
    #计算判别器的误差函数
    #采样生成图片
    fake_image=generator(batch_z,is_training)
    #判定生成图片
    d_fake_logits=discriminator(fake_image,is_training)
    #判定真实图片
    d_real_logits=discriminator(batch_x,is_training)
    #真实图片与1之间的误差
    d_loss_real=celoss_ones(d_real_logits)
    #生成图片与0之间的误差
    d_loss_fake=celoss_zeros(d_fake_logits)
    #合并误差
    loss=d_loss_fake+d_loss_real
    
    return loss

#%%
'''
生成网络

由于真实样本与生成器无关，
因此误差函数只需要考虑最小化𝔼𝒛~𝑝𝑧(∙)log(1 − 𝐷𝜃(𝐺𝜙(𝒛)))项即可。
可以通过将生成的样本标 注为 1，最小化此时的交叉熵误差。
需要注意的是，在反向传播误差的过程中，判别器也参与了计算图的构建，
但是此阶段只需要更新生成器网络参数，而不更新判别器的网络参数。

'''

def g_loss_fn(generator,discriminator,batch_z,is_training):
    #采样生成图片
    fake_image=generator(batch_z,is_training)
    #训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits=discriminator(fake_image,is_training)
    #计算生成图片与1之间的误差
    loss=celoss_ones(d_fake_logits)
    
    return loss

#%%

'''
网络训练

在每个 Epoch，首先从先验分布𝑝 (∙)中随机采样隐藏向量，
从真实数据集中随机采样真实图片，
通过生成器和判别器计算判别器网络的损失，
并优化判别器网络参数𝜃。
在训练生成器时，需要借助于判别器来计算误差，但是只计算生成器的梯度信息并更新𝜙。
这里设定判别器训练𝑘 = 5次后，生成器训练一次。
'''

#首先创建生成网络和判别网络，并分别创建对应的优化器

def main():
    z_dim=100#隐藏变量z的维度
    
    generator=Generator()#创建生成器
    generator.build(input_shape=(batch_size,z_dim))
    generator.summary()
    discriminator=Discriminator()#创建判别器
    discriminator.build(input_shape=(None,64,64,3))
    discriminator.summary()
    #分别为生成器和判别器创建优化器
    
    if os.path.exists('checkpoint')==True:
        generator.load_weights('generator.ckpt')
        print('Loaded generator.ckpt!')
        discriminator.load_weights('discriminator.ckpt')
        print('Loaded discriminator.ckpt!')

    learning_rate=0.0002
    g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.5)
    d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.5)
    
    epochs=100000#会一定间隔后保存权重，所以epoch可以无限
    db_iter = iter(dataset)
    
    is_training=True
        
    #主训练部分代码实现如下
    for epoch in range(epochs):#训练epochs次
        #1.训练判别器
        batch_z=tf.random.normal([batch_size,z_dim])
        batch_x=next(db_iter)#采用真实照片
        #判别器前向计算
        with tf.GradientTape() as tape:
            d_loss=d_loss_fn(generator,discriminator,batch_z,batch_x,is_training)
        grads=tape.gradient(d_loss,discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))
        #2.训练生成器
        #采样隐藏变量
        for _ in range(2):
	        batch_z=tf.random.normal([batch_size,z_dim])
	        with tf.GradientTape() as tape:
	            g_loss=g_loss_fn(generator,discriminator,batch_z,is_training)
	        grads=tape.gradient(g_loss,generator.trainable_variables)
	        g_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
        
        #每间隔100个Epoch，进行一次图片生成测试。
        #通过从先验分布中随机采样隐向量，送入 生成器获得生成图片，并保存为文件。
        
        if epoch % 100 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss))
            # 可视化
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('gan_images', 'gan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')
    
            if epoch>0 and epoch % 1000 == 0:
                generator.save_weights('generator.ckpt')
                discriminator.save_weights('discriminator.ckpt')
                
                if epoch%5000==0:
                    learning_rate/=2
                    g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.5)
                    d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.5)

def save_result(val_out, val_block_size, image_path, color_mode):
    '''
    Parameters
    ----------
    val_out : [100,64,64,3]的生成图片
    val_block_size : 每行每列val_block_size个图片
    image_path : 图片存储路径
    color_mode : TYPE
    '''
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)
            #在[h,w,c]的w维度插入

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
                #在[h,w,c]的h维度插入
            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        #如果c通道为1，就保留1
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)

    
#%%
'''
生成单张图片
'''
import random

def product():
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img
    
    z_dim=100
    Input=tf.random.normal([1,z_dim])
    generator=Generator()#创建生成器
    generator.load_weights('generator.ckpt')
    print('Loaded generator.ckpt!')
    Output=generator(Input,training=False)
    del generator
    image_path=str(random.randint(0,int(1e9)))+'.png'
    Image.fromarray(np.array(preprocess(Output[0,:,:,:].numpy()))).save(image_path)
    
#%%
    
if __name__=='__main__':
    main()
    
    
