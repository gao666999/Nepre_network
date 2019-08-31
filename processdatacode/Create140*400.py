import numpy as np
import os
import tensorflow as tf
from fnmatch import fnmatchcase as match
def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def CreateDataMatrix():
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

def TurnDictToArray(file):
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
    return DataArray

def create_tfrecords(npypath,TFpath,savefilename):
    #filename = '/public/home/xgao/Desktop/CreateData/outputtrain.tfrecords'
    savefile = os.path.join(TFpath,savefilename)
    writer = tf.python_io.TFRecordWriter(savefile)
    filenames = ListdirInMac(npypath)
    filenames.sort()
    for filename in filenames:
        print (filename)
        file = os.path.join(npypath,filename)
        data = TurnDictToArray(file)
        data = data.astype(float)
        allsum = sum(sum(data))
        #data = data/allsum
        print (allsum)
        if allsum != 0:
            data = data/allsum
            data = data.flatten()
            data = data*100
            '''
            if match(filename,'*P.npy') and allsum>40:
                label = np.array([1,0])
                example=tf.train.Example(features=tf.train.Features(feature={
                    'data':tf.train.Feature(float_list=tf.train.FloatList(value=data)),
                    'label':tf.train.Feature(float_list=tf.train.FloatList(value=label))}))
                writer.write(example.SerializeToString())
                print('finishedP')
            ''' 
            if match(filename,'*N.npy') and allsum >100:
                label = np.array([0,1])
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'data':tf.train.Feature(float_list = tf.train.FloatList(value = data)),
                    'label':tf.train.Feature(float_list = tf.train.FloatList(value = label))}))
                writer.write(example.SerializeToString())
                print('finishedN')
          
        #except:
            #pass
    writer.close()

if __name__ == "__main__":
    datapath = '/public/home/xgao/Desktop/zdocktest/PNdicttest/'
    TFpath = '/public/home/xgao/Desktop/CleanZdock/usefulTFdata/ValidTF/'
    savefilename = 'Zdocktest.tfrecords'
    create_tfrecords(datapath,TFpath,savefilename)



