import numpy as np
import sys
import os

args = sys.argv[1:]
filename = args[0]
path = '/public/home/xgao/Desktop/CreateData/PNarraytrain'
file = os.path.join(path,filename)
data = np.load(file)
for i in range(0,20):
    print (data[i])
#print (data)

