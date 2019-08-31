import os
import math
import numpy as np
import random
from mpi4py import MPI
comm = MPI.COMM_WORLD
mrank = comm.Get_rank()
msize = comm.Get_size()


def ListdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def ExtractData(line):
    """
    This part will extracted data from line according to the standard
    PDB file format(Version 3.3.0, Nov.21, 2012)
    """
    res = []

    line = line.strip()
    #record_name
    res.append(line[0:4].strip(' '))

    #atom_serial
    res.append(line[6:11].strip(' '))

    #atom_name
    res.append(line[12:16].strip(' '))

    #alternate_indicator
    res.append(line[16])

    #residue_name
    res.append(line[17:20].strip(' '))

    #chain_id
    res.append(line[21].strip(' '))

    #residue_num
    res.append(line[22:26].strip(' '))

    #xcor
    res.append(line[30:38].strip(' '))

    #ycor
    res.append(line[38:46].strip(' '))

    #zcor
    res.append(line[46:54].strip(' '))

    return res


def ProcessPdb(pdbfile):
    Currentresidue_num = None
    AllResidueList = []
    AllReTERList = []
    Residue = []
    TERlist = []
    with open(pdbfile) as f:
        numTER = 0
        for line in f.readlines():
            if(line[0:4] == 'ATOM'):
                element_list = ExtractData(line)
                residue_num = element_list[-4]
                if (Currentresidue_num is None):
                    #print 'nnnnnnnnnnnnnn'
                    Currentresidue_num = residue_num
                    #print line,'kkkkkkkkkkkkkkkkkkkks'
                    Residue.append(line)
                elif(residue_num != Currentresidue_num):
                    #print residue_num
                    #print 'kkkkkkkkkkkkkkkkkkkks'
                    AllReTERList.append(Residue)
                    AllResidueList.append(Residue)
                    Residue =[]
                    Currentresidue_num = residue_num
                    #print line,'nnnnnnnnnnnnnn'
                    Residue.append(line)
                elif(residue_num == Currentresidue_num):
                    Currentresidue_num = residue_num
                    #print line,'xxxxxxxxxxxxxxxxxx'
                    Residue.append(line)
            elif(line[0:3] == 'TER'):
                #print'rrrrrrr'
                TERlist.append(line)
                AllReTERList.append(TERlist)
                TERlist = []
                numTER +=1
            elif numTER == 2:
                break
    return AllResidueList,AllReTERList


def ReplaceAAname(Residuecurrent,ResidueNext):
    NewResidue =[]
    AAname = ResidueNext[0][17:20].strip(' ')
    for line in Residuecurrent:
        linenew = line[:17]+ AAname + line[20:]
        NewResidue.append(linenew)
    return NewResidue

'''
def CreateNewpdb(originalpdb,newpdb):
    #print 'hhhhhh'
    AllResidueList = ProcessPdb(originalpdb)
    #print AllResidueList
    with open(newpdb,'w') as newf:
        length = len(AllResidueList)
        for r in range(length):
            Residuecurrent = AllResidueList[r]
            #print Residuecurrent[0][:4]
            if Residuecurrent[0][:4] == 'ATOM':
                a = random.randint(0,length-1)
                #print AllResidueList[a]
                #print 'ffffffffffffffffff'
                while AllResidueList[a][0][:3] == 'TER':
                    a = random.randint(0,length-1)
                ResidueNext = AllResidueList[a]
                NewResidue = ReplaceAAname(Residuecurrent,ResidueNext)
                for linereplaced in NewResidue:
                    #print linereplaced
                    newf.write(linereplaced)
            elif Residuecurrent[0][:3] == 'TER':
                print 'tttttttttttttttttttttt'
                lineTER = Residuecurrent[0]
                newf.write(lineTER)
'''

def CreateNewpdb(originalpdb,newpdb):
    #print 'hhhhhh'
    AllResidueList,AllReTERList = ProcessPdb(originalpdb)
    #print AllResidueList
    with open(newpdb,'w') as newf:
        lengthr = len(AllResidueList)
        lengthrt = len(AllReTERList)
        arr = np.arange(lengthr)
        np.random.shuffle(arr)
        n = 0
        for r in range(lengthrt):
            Residuecurrent = AllReTERList[r]
            #print Residuecurrent[0][:4]
            if Residuecurrent[0][:4] == 'ATOM':
                #if r >=lengthr:
                a = arr[n]
                n+=1
                ResidueNext = AllResidueList[a]
                NewResidue = ReplaceAAname(Residuecurrent,ResidueNext)
                for linereplaced in NewResidue:
                    #print linereplacedCre
                    newf.write(linereplaced)
            elif Residuecurrent[0][:3] == 'TER':
                #print 'tttttttttttttttttttttt'
                lineTER = Residuecurrent[0]
                newf.write(lineTER)



def SampleNegative(filenames,pdbpath,newpdbpath):
    for filename in filenames:
        originalpdb = os.path.join(pdbpath,filename)
        newpdb = os.path.join(newpdbpath,filename)
        CreateNewpdb(originalpdb,newpdb)


def MultiprocessRun(filepath,newpdbpath):
    allnames = ListdirInMac(filepath)
    filelength = len(allnames)
    processnum = 120
    perSize = int(math.ceil(float(filelength)/float(processnum)))
    allfilenames = []
    for process_n in range(processnum):
        if perSize*(process_n + 1) > len(allnames):
            rightside = len(allnames)
            leftside = perSize*process_n
        else:
            leftside = perSize * process_n
            rightside = perSize * (process_n + 1)
        #print leftside,rightside,filelength
        filenames = allnames[leftside:rightside]
        allfilenames.append(filenames)

    for i,filenames in enumerate(allfilenames):
        if i % msize != mrank:
            continue
        SampleNegative(filenames,filepath,newpdbpath)


if __name__ == "__main__":
    '''
    filenames = ['3wxy.pdb']
    filepath = '/Users/xg666/Downloads/'
    newpdbpath = '/Users/xg666/Desktop/NeChmachinelearn/negativesample'
    SampleNegative(filenames,filepath,newpdbpath)
    '''
    filepath = '/public/home/xgao/Desktop/CreateData/positivesample/'
    newpdbpath = '/public/home/xgao/Desktop/CreateData/negativesample/'
    MultiprocessRun(filepath,newpdbpath)

