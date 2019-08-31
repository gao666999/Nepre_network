import numpy as np
dict3mp7=np.load('/home/xg666/Desktop/zdocktrain/Vdict/3mp7dict.npy').item()
keys= dict3mp7.keys()
for key in keys:
    print (key,dict3mp7[key])
