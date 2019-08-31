import tensorflow as tf
import numpy as np
import os
from fnmatch import fnmatchcase as match
def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def _int8_feature(value):
    return tf.train.Feature(int8_list = tf.train.Int8List(value = [value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def create_tfrecords(trainpath):
    filename = '/public/home/xgao/Desktop/CreateData/outputtrain.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    filenames = ListdirInMac(trainpath)
    for filename in filenames:
        print (filename)
        file = os.path.join(trainpath,filename)
        #data = np.load(file)
        #print (data)
        #data = data.flatten()
        #print (data)
        
        #try:
        data = np.load(file)
        #print (data)
        data = data.flatten()
        #print (data)
        if match(filename,'*P.npy'):
            lable = np.array([1,0])
        if match(filename,'*N.npy'):
            lable = np.array([0,1])
        #print('jjjjjjjjjjjjjjj')
        example = tf.train.Example(features = tf.train.Features(feature = {
            'data':tf.train.Feature(float_list = tf.train.FloatList(value = data)),
            'lable':tf.train.Feature(float_list = tf.train.FloatList(value = lable))
            }))
        #print('vvvvvvv')
        writer.write(example.SerializeToString())
        #print('wwwwww')   
        #except:
            #pass
    writer.close()
if __name__=='__main__':
    #find_biggest()
    trainpath = '/public/home/xgao/Desktop/CreateData/PNarraytrain'
    testpath = '/public/home/xgao/Desktop/CreateData/PNarraytest'
    create_tfrecords(trainpath)
