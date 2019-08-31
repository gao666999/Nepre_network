#coding=utf-8

import tensorflow as tf
import numpy as np
#import pdb
import os
from datetime import datetime
import slim.nets.resnet_v1 as resnet_v1
#from create_tf_record import *
import tensorflow.contrib.slim as slim

'''
参考资料：https://www.cnblogs.com/adong7639/p/7942384.html
'''
labels_nums = 2  # 类别个数
batch_size = 100
resize_height = 140  # mobilenet_v1.default_image_size 指定存储图片高度
resize_width = 400   # mobilenet_v1.default_image_size 指定存储图片宽度
depths = 1
data_shape = [batch_size, resize_height, resize_width, depths]
val_nums = 3000
#MODEL_SAVE_PATH='/home/xg666/Desktop/zdocktrain/savemodels/resnet_140/'
#folder_of_dataset = '/home/xg666/Desktop/NECNN/TFfiles/'
# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')
folder_of_dataset='/home/xg666/Desktop/zdocktrain/usefulTFdata/ValidTF/'
def read_tfrecords(filename):
    filename_quene=tf.train.string_input_producer([filename])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_quene)
    features=tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([2], tf.float32),
                                      'data' : tf.FixedLenFeature([56000], tf.float32)
                                      })
    traindata=tf.reshape(features['data'],(140,400,1))
    labeldata =features['label']
    return traindata,labeldata

def get_batch_images(tffilename):
  with tf.device('/cpu:0'):
    train_set = os.path.join(folder_of_dataset,tffilename)
    #test_set = os.path.join(folder_of_dataset,'outputtest.tfrecords')
    traindata,trainlabel = read_tfrecords(train_set)
    capacity = 2*batch_size
    train_data,train_label = tf.train.shuffle_batch([traindata,trainlabel],batch_size = batch_size,capacity = capacity,min_after_dequeue = 100)
    return train_data,train_label

def net_evaluation(sess,accuracy,val_images_batch,val_labels_batch):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_acc = sess.run(accuracy, feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_accs.append(val_acc)
    #mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_acc


def train(val_record_file,
          labels_nums,
          data_shape,
          snapshot_prefix):
    '''
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step: 显示训练过程log信息间隔
    :param train_param: train参数
    :param val_record_file: 验证的tfrecord文件
    :param val_log_step: 显示验证过程log信息间隔
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_shape: 输入数据shape
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    #[base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    val_images_batch, val_labels_batch = get_batch_images(val_record_file)
    # Define the model:
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        out, end_points = resnet_v1.resnet_v1_101(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    saver=tf.train.Saver()
    while True:
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            ckpt = tf.train.get_checkpoint_state(snapshot_prefix)
            if ckpt and ckpt.model_checkpoint_path:
              saver.restore(sess,ckpt.model_checkpoint_path)
              global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
              print(global_step)
              coord = tf.train.Coordinator()
              threads = tf.train.start_queue_runners(sess=sess, coord=coord)
              #batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
              mean_acc = net_evaluation(sess, accuracy, val_images_batch, val_labels_batch)
              print("%s: val accuracy :  %g" % (datetime.now(), mean_acc))
              coord.request_stop()
              coord.join(threads)
'''
def eavlate():
    is_train
'''
if __name__ == '__main__':
    #train_record_file='0.05train.tfrecords'
    val_record_file='1tks.tfrecords'
    #print('1ugh')
    snapshot_prefix='/home/xg666/Desktop/zdocktrain/savemodels/resnet_140/'
    train(val_record_file=val_record_file,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot_prefix=snapshot_prefix)
