import os
from fnmatch import fnmatchcase as match
path='/home/xg666/Desktop/zdocktrain/PNnpy/3mp7'
filenames=os.listdir(path)
for name in filenames:
    if match(name,'*P.npy'):
       print(name)
