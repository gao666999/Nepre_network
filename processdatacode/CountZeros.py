import os
import numpy as np

def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list
def CoutCount(path):
    filenames = ListdirInMac(path)
    length = len(filenames)
    count = 0
    for filename in filenames:
        print (filename)
        file = os.path.join(path,filename)
        AAdict = np.load(file).item()
        aaList=["ALA","VAL","LEU","ILE","PHE","TRP","MET","PRO","GLY","SER",
            "THR","CYS","TYR","ASN","GLN","HIS","LYS","ARG","ASP","GLU"]
        Array = np.zeros([20,20])
        for i in range(len(aaList)):
            Amino1 = aaList[i]
            for j in range(len(aaList)):
                Amino2 = aaList[j]
                matrix = AAdict[Amino1][Amino2]
                value = sum(sum(matrix))
                Array[i][j] = value
        summatrix = sum(sum(Array))
        print (summatrix)
    if int(summatrix) != 0:
            count+=1
    print (length,count,float(count)/float(length))
if __name__ == "__main__":
    path = '/public/home/xgao/Desktop/CreateData/negativedata'
    CoutCount(path)

