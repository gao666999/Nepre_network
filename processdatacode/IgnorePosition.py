import os
from random import shuffle
import numpy as np
import tensorflow as tf
from fnmatch import fnmatchcase as match
def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list
def Sumposition(matrix):
    value = sum(sum(matrix))
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
        
    
    return Sumall,allsum,Array 



def create_tfrecords(npypath,TFpath,savefilename):
    #filename = '/public/home/xgao/Desktop/CreateData/outputtrain.tfrecords'
    savefile = os.path.join(TFpath,savefilename)
    writer = tf.python_io.TFRecordWriter(savefile)
    filenames = ListdirInMac(npypath)
    count = 0
    length = len(filenames)
    shuffle(filenames)
    for filename in filenames:
        #print (filename)
        file = os.path.join(npypath,filename)
        try:
            index,allsum,data = DictToArray(file)
            print (allsum,filename)
            if index == 1:
                #print ('no zore')
                count += 1
                data = data.flatten()
                '''
                if match(filename,'*P.npy') and allsum >54 and allsum<500:
                    label = np.array([1,0])
                    print ('ppppppp',filename,label)
                    exmple=tf.train.Example(features=tf.train.Features(feature={
                          'data':tf.train.Feature(float_list=tf.train.FloatList(value=data)),
                          'label':tf.train.Feature(float_list=tf.train.FloatList(value=label))}))
                    writer.write(example.SerializeToString())
                    print(label,'finished')
                '''
                if match(filename,'*P.npy') and allsum >40:
                    label = np.array([1,0])
                    print('PPPPPPPP',filename,label)
                    example = tf.train.Example(features = tf.train.Features(feature = {
                        'data':tf.train.Feature(float_list = tf.train.FloatList(value = data)),
                        'label':tf.train.Feature(float_list = tf.train.FloatList(value = label))}))
                    writer.write(example.SerializeToString())
                    print('finishedP')
                if match(filename,'*N.npy') and allsum>100:
                    label=np.array([0,1])
                    print('NNNNN',filename,label)
                    example=tf.train.Example(features=tf.train.Features(feature={
                        'data':tf.train.Feature(float_list=tf.train.FloatList(value=data)),
                        'label':tf.train.Feature(float_list=tf.train.FloatList(value=label))}))
                    writer.write(example.SerializeToString())
                    print('finishedN')
        except:
            pass
    print (count,length)
    writer.close()

if __name__ == "__main__":
    datapath = '/public/home/xgao/Desktop/CreateZdock/PNtest'
    TFpath = '/public/home/xgao/Desktop/CleanZdock/usefulTFdata/20TF/test/'
    savefilename = '20test.tfrecords'
    create_tfrecords(datapath,TFpath,savefilename)


