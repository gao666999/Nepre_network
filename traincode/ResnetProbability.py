#coding=utf-8
import tensorflow as tf
import numpy as np
#import pdb
import os
from datetime import datetime
import slim.nets.resnet_v1 as resnet_v1
#from create_tf_record import *
import tensorflow.contrib.slim as slim
import sys
from fnmatch import fnmatchcase as match
#import time
#EVAL_INTERVAL_SECS = 10
MODEL_SAVE_PATH = '/home/xg666/Desktop/zdocktrain/savemodels/resnet_20/'

def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def DictToArray(file):
    AAdict = np.load(file,encoding = 'latin1').item()
    aaList=["ALA","VAL","LEU","ILE","PHE","TRP","MET","PRO","GLY","SER",
            "THR","CYS","TYR","ASN","GLN","HIS","LYS","ARG","ASP","GLU"]
    Array = np.zeros([20,20])
    Sumall = int(0)
    for i in range(len(aaList)):
        Amino1 = aaList[i]
        for j in range(len(aaList)):
            Amino2 = aaList[j]
            matrix = AAdict[Amino1][Amino2]
            value = sum(sum(matrix))
            #print (matrix)
            #print (value)
           # print ('jjjjjjjjj')
            Array[i][j] = value
    allsum = int(sum(sum(Array)))
    if allsum != 0:
        Sumall = 1
        print (allsum)
        Array = Array.astype(float)
        Array = Array/allsum
        #print (Array,allsum)
        Array = Array.flatten()
    return Sumall,Array


def GetDate(path,recordfile):
    filenumber = 0
    with open (recordfile,'w') as fileR:
        filenames = ListdirInMac(path)[:1000]
        inputdata = np.zeros((0,400))
        for filename in filenames:
            if match(filename,'complex*'):
                file = os.path.join(path,filename)
                index,data = DictToArray(file)
                if index == 1:
                    filenumber += 1
                    inputdata = np.row_stack((inputdata,data))
                    fileR.write(filename +str('\n'))
    return inputdata,filenumber



def eavluate(path,recordfile,resultfile):
    inputdata,arraylength = GetDate(path,recordfile)
    inputdata = tf.convert_to_tensor(inputdata)
    inputdata = tf.reshape(inputdata,(arraylength,20,20,1))
    inputdata = tf.cast(inputdata,dtype=tf.float32)
    is_training = False
    labels_nums = 2
    keep_prob=1
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        out, end_points = resnet_v1.resnet_v1_101(inputs=inputdata, num_classes=labels_nums, is_training=is_training,global_pool=True)
    outputprobability = tf.nn.softmax(out)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.get_variable_scope().reuse_variables()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            probabilitylist = sess.run(outputprobability)
            probabilityresult = probabilitylist
            print (type(probabilitylist))
            print (probabilityresult)
            np.savetxt(resultfile,probabilityresult)
            #print (accuracy_score_test)
            #print (type(accuracy_score_test))
            #print ('After %s training step(s),validation''test_accury = %g'%(global_step,accuracy_score_test))
            #print ('After %smtraining step(s),validation''train_accury = %g'%(global_step,accuracy_score_train))
        else:
            print('No checkpoint file found')
            return

def train():
    args = sys.argv[1:]
    PDBname = args[0]
    #PDBname = '1UDI'
    path = os.path.join('/home/xg666/Desktop/NECNN/zdocknpydata/',PDBname)
    resultfile = os.path.join('/home/xg666/Desktop/NECNN/zdockresult/R20',str(PDBname)+str('result.txt'))
    recordfile = os.path.join('/home/xg666/Desktop/NECNN/zdockrecord/R20',str(PDBname)+str('record.txt'))
    eavluate(path,recordfile,resultfile)
if __name__=='__main__':
    #tf.app.run()
    train()





