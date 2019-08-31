import numpy as np
import sys
import os

def listdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def Datamain(file):
    data = np.load(file,encoding='latin1').item()
 #print (data)
    if data ==None:
        print ('nnnnnnn')
    AAList = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    Array = np.zeros(shape = (0,400))
    for amino1 in AAList:
        for amino2 in AAList:
            value = data[amino1][amino2].flatten()
            print (sum(value))
            Array = np.row_stack((Array,value))
    valuenum=sum(sum(Array))
    #if valuenum>0:
    return valuenum
    #print (sum(sum(Array)))
if __name__=='__main__':
    file = '/public/home/xgao/Desktop/CleanZdock/Vpositivenpy/1baj/complex.1.npy'
    valuenum = Datamain(file)
    print (valuenum)
   #args = sys.argv[1:]
   #filename = args[0]
   # numberlist=[]
   # path = '/public/home/xgao/Desktop/CreateZdock/Pnpy'
   # filenames=listdirInMac(path)
   # allsum=0
   # samplenum=0
   # for filename in filenames:
   #     file = os.path.join(path,filename)
   #     #print (file)
   #     valuenum=Datamain(file)
   #     if valuenum>0:
   #         print(valuenum)
   #         numberlist.append(valuenum)
   #         allsum+=valuenum
   #         samplenum+=1
   # average=allsum/float(samplenum)
   # print (average,samplenum,allsum)
   # numberresult=np.array(numberlist)
   # savefile='./positivenumber.npy'
   # np.save(savefile,numberresult)
