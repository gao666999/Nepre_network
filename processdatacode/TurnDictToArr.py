import numpy as np
import os
#from funct
def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
            #print b
    return b

def savefile(file,result):
    savefile = file
    np.save(savefile,result)

def Datamain(file,filenpy):
    data = np.load(file).item()
    #print data
    AAList = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', \
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    Array = np.zeros(shape = (0,400))
    for amino1 in AAList:
        for amino2 in AAList:
            #print amino1,amino2
            #print data[amino1][amino2]
            value = data[amino1][amino2].flatten()
            #value = value.astype(int)
            #print value
            #value = np.array(flatten(values))
            #print type(value)
            Array = np.row_stack((Array,value))
    #print Array.shape
    savefile(filenpy,Array)
def LoadMain(path,resultpath):
    filenames = ListdirInMac(path)
    print len(filenames)
    n = 0
    for filename in filenames:
        n+=1
        file = os.path.join(path,filename)
        newname = filename[:4]+str('P.npy')
        newfile = os.path.join(resultpath,newname)
        Datamain(file,newfile)

if __name__ == "__main__":
    '''
    path = '/public/home/xgao/Desktop/NeChlearn/positivedatatwo'
    resultpath = '/public/home/xgao/Desktop/NeChlearn/Pdatatxt'
    LoadMain(path,resultpath)
    '''
    path = '/public/home/xgao/Desktop/CreateData/positivedata'
    resultpath = '/public/home/xgao/Desktop/CreateData/Parray'
    LoadMain(path,resultpath)

