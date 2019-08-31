import tensorflow as tf
#import CNN_inference
#import CNN_202 as CN
import slim.nets.mobilenet_v1 as mobilenet_v1
import tensorflow.contrib.slim as slim
import os
import numpy as np
import sys
from fnmatch import fnmatchcase as match
#import time
#EVAL_INTERVAL_SECS = 10
#MODEL_SAVE_PATH = '/home/xg666/Desktop/TF_classification_models/tensorflow_models_learning-master/mobilenet_models_zdock_0603'
MODEL_SAVE_PATH = '/home/xg666/Desktop/zdocktrain/savemodels/mobilenet_140/'
def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list


def createDataMatrix():
    #TYR and MET belong to two catergories at the same time
    aaList=["ALA","VAL","LEU","ILE","PHE","TRP","MET","PRO","GLY","SER",
            "THR","CYS","TYR","ASN","GLN","HIS","LYS","ARG","ASP","GLU"]

    Charged = ['ARG','LYS','ASP','GLU']
    Polar = ['GLN','ASN','HIS','SER','THR','CYS']
    Amphipathic = ['TRP','TYR','MET']
    Hydrophobic = ['ALA','ILE','LEU','PHE','VAL','PRO','GLY']

    Charged_add = ['ARG','LYS','ASP','GLU','GLN','ASN','HIS']
    Polar_add = ['GLN','ASN','HIS','SER','THR','TYR','CYS']
    Amphipathic_add = ['TRP','TYR','MET','GLN','ASN','HIS','SER']
    Hydrophobic_add = ['ALA','ILE','LEU','PHE','VAL','PRO','GLY']


def JudgeCategory(Amnio):
    #category = None
    categorys = ['C','P','A','H']
    AAcategorys = [['ARG','LYS','ASP','GLU'],
    ['GLN','ASN','HIS','SER','THR','CYS'],
    ['TRP','TYR','MET'],
    ['ALA','ILE','LEU','PHE','VAL','PRO','GLY']]
    for i in range(len(AAcategorys)):
        for j in range(len(AAcategorys[i])):
            if Amnio == AAcategorys[i][j]:
                break
            else:
                continue
    category = categorys[i]
    return category

def DictToArray(file):
    AAdict = np.load(file,encoding = 'latin1').item()
    aaList=["ALA","VAL","LEU","ILE","PHE","TRP","MET","PRO","GLY","SER",
            "THR","CYS","TYR","ASN","GLN","HIS","LYS","ARG","ASP","GLU"]
    Charged_add = ['ARG','LYS','ASP','GLU','GLN','ASN','HIS']
    Polar_add = ['GLN','ASN','HIS','SER','THR','TYR','CYS']
    Amphipathic_add = ['TRP','TYR','MET','ALA','ILE','LEU','PHE']
    Hydrophobic_add = ['ALA','ILE','LEU','PHE','VAL','PRO','GLY']
    DataArray = np.zeros((0,400))

    for Amino1 in aaList:
        category = JudgeCategory(Amino1)
        if category == 'C':
            for Amino2 in Charged_add:
                row = AAdict[Amino1][Amino2].flatten()
                DataArray = np.row_stack((DataArray,row))
        elif category == 'P':
            for Amino2 in Polar_add:
                row = AAdict[Amino1][Amino2].flatten()
                DataArray = np.row_stack((DataArray,row))
        elif category == 'A':
            for Amino2 in Amphipathic_add:
                row = AAdict[Amino1][Amino2].flatten()
                DataArray = np.row_stack((DataArray,row))
        elif category == 'H':
            for Amino2 in Hydrophobic_add:
                row = AAdict[Amino1][Amino2].flatten()
                DataArray = np.row_stack((DataArray,row))
    allsum = int(sum(sum(DataArray)))
    if allsum != 0:
        Sumall = 1
        print (allsum)
        DataArray = DataArray.astype(float)
        DataArray = DataArray/allsum
        #print (Array,allsum)
        DataArray = DataArray.flatten()
    return Sumall,DataArray,allsum
    #return DataArray
'''
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
    return Sumall,Array,allsum
'''

def GetDate(path,recordfile):
    filenumber = 0
    with open (recordfile,'w') as fileR:
        filenames = ListdirInMac(path)[:1000]
        inputdata = np.zeros((0,56000))
        for filename in filenames:
            if match(filename,'complex*'):
                file = os.path.join(path,filename)
                index,data,allsum = DictToArray(file)
                if 50<allsum:
                    filenumber += 1
                    inputdata = np.row_stack((inputdata,data))
                    fileR.write(filename +str('\n'))
    return inputdata,filenumber





def eavluate(path,resultfile,recordfile):
    '''
    with tf.device('/cpu:0'):
        test_set = os.path.join(CN.folder_of_dataset,'ignorepositiontest.tfrecords')
        capacity = 1000+3*CN.BATCH_SIZE
        test_data,test_label = CN.read_tfrecords(test_set)
        xtest,y_test = tf.train.shuffle_batch([test_data,test_label],batch_size = 20000,capacity = capacity,min_after_dequeue = 30)
    '''
        #train_set = os.path.join(CNN_train2.folder_of_dataset,'140*400train.tfrecords')
        #capacity = 1000+3*CNN_train2.BATCH_SIZE
        #train_data,train_label = CNN_train2.read_tfrecords(train_set)
        #xtrain,y_train = tf.train.shuffle_batch([train_data,train_label],batch_size = CNN_train2.BATCH_SIZE,capacity = capacity,min_after_dequeue = 30)

    '''
    with tf.Graph() as_default() as g:
        x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
        y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')
    validate_feed = {x:testdata,y_:testlabel}
    '''
    inputdata,arraylength = GetDate(path,recordfile)
    inputdata = tf.convert_to_tensor(inputdata)
    inputdata = tf.reshape(inputdata,(arraylength,140,400,1))
    inputdata = tf.cast(inputdata,dtype=tf.float32)
    train = False
    labels_nums = 2
    keep_prob=1
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        out,end_points=mobilenet_v1.mobilenet_v1(inputs=inputdata,num_classes=labels_nums,dropout_keep_prob=keep_prob,is_training=train,global_pool=True)
    #ytest = CNN_inference.inference(inputdata,train,None)
    outputprobability = tf.nn.softmax(out)
    #ytrain = CNN_inference.inference(xtrain,train,None)
    #correct_prediction_test = tf.equal(tf.argmax(ytest,1),tf.argmax(y_test,1))
    #correct_prediction_train = tf.equal(tf.argmax(ytrain,1),tf.argmax(y_train,1))
    #accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test,tf.float32))
    #accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train,tf.float32))
    #variable_averages = tf.train.ExponentialMovingAverage(0.99)
    #variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()
    #resultfile = './result.txt'
    #while True:
    with tf.Session() as sess:
        tf.get_variable_scope().reuse_variables()
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #print ('**************************')
            #print (global_step)
            #global_step = tf.cast(global_step,dtype=tf.int32)
            #global_step
            #print (type(global_step))
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
        #return probabilitylist

def train(PDBname):
    #args = sys.argv[1:]
    #PDBname = args[0]
    path = os.path.join('/home/xg666/Desktop/NECNN/zdocknpydata/',PDBname)
    resultfile = os.path.join('/home/xg666/Desktop/NECNN/zdockresult/M140',str(PDBname)+str('result.txt'))
    recordfile = os.path.join('/home/xg666/Desktop/NECNN/zdockrecord/M140',str(PDBname)+str('record.txt'))
    eavluate(path,resultfile,recordfile)

    #return probabilitylist

if __name__=='__main__':
    #tf.app.run()
    args = sys.argv[1:]
    PDBname = args[0]
    #PDBname = '1FBI'
    train(PDBname)


