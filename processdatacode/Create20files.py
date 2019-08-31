import numpy as np
import os
import tensorflow as tf
import math
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
       # print (allsum)
        Array = Array.astype(float)
        Array = Array/allsum
       # print (Array,allsum)
    return Sumall,Array,allsum

def create_tfrecords(npypath,filenames,TFpath,savefilename):
    #filename = '/public/home/xgao/Desktop/CreateData/outputtrain.tfrecords'
    savefile = os.path.join(TFpath,savefilename)
    writer = tf.python_io.TFRecordWriter(savefile)
    #filenames = ListdirInMac(npypath)
    count = 0
    length = len(filenames)
    filenames.sort()
    for filename in filenames:
        #print (filename)
        file = os.path.join(npypath,filename)
        try:
            index,data,allsum = DictToArray(file)
           # print (index)
            if index == 1:
                print ('no zore')
                count += 1
                data = data.flatten()
                #print(data)
                #data = data/float(allsum)
                data=data*100
                #print (data)
                #print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
                if match(filename,'*P.npy') and allsum >40 :
                    label = np.array([1,0])
                    example=tf.train.Example(features=tf.train.Features(feature={
                        'data':tf.train.Feature(float_list=tf.train.FloatList(value=data)),
                        'label':tf.train.Feature(float_list=tf.train.FloatList(value=label))}))
                    writer.write(example.SerializeToString())
                    print('finishedP')
                if match(filename,'*N.npy') and allsum >100:
                    label = np.array([0,1])
                    example = tf.train.Example(features = tf.train.Features(feature = {
                        'data':tf.train.Feature(float_list = tf.train.FloatList(value = data)),
                        'label':tf.train.Feature(float_list = tf.train.FloatList(value = label))}))
                    writer.write(example.SerializeToString())
                    print('finishedN')
        except:
            pass
    #print (count,length)
    
    writer.close()
    return count

def CreateFiles(datapath,TFpath):
    alldatanames=ListdirInMac(datapath)
    TFfileNum=10
    length=len(alldatanames)
    perSize = int(math.ceil(float(length)/float(TFfileNum)))
    num_sample=0
    for process_n in range(10):
        savefilename=str('20*20_')+str(process_n)+str('train.tfrecords')
        if perSize*(process_n + 1) > length:
            rightside = length
            leftside = perSize*process_n
        else:
            leftside = perSize * process_n
            rightside = perSize * (process_n + 1)
        print (leftside,rightside)
        filenames=alldatanames[leftside:rightside]
        count=create_tfrecords(datapath,filenames,TFpath,savefilename)
        num_sample+=count
        print (num_sample)

if __name__ == "__main__":
    datapath = '/public/home/xgao/Desktop/CreateZdock/PNtrain/'
    TFpath = '/public/home/xgao/Desktop/CleanZdock/usefulTFdata/20TF/train/'
    #savefilename = '140*400zdocktest0.05one.tfrecords'
    #create_tfrecords(datapath,TFpath,savefilename)
    CreateFiles(datapath,TFpath)


