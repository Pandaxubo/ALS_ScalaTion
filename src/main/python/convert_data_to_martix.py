### author: Zhichen Yan
### This file is used to convert original data(MovieLens 100k format) to rating matrix.
### For former experiments use
import numpy as np
import pandas as pd

### open the original document
f = open ( '500data.txt' , 'r')

### split data and get the rows and columns we want
data = np.array([line.split() for line in f])
print(data.shape)
rn = np.array(data[:,0]) #row array
rn = np.unique(rn.astype(np.int)).astype(np.str)#row name
rns = np.unique(rn).size #row size
print(rns)
cn = np.array(data[:,1]) 
cn = np.unique(cn.astype(np.int)).astype(np.str)
cns = cn.size
print(cns)

### construct output list
tar = np.zeros([rns,cns])
print(tar)
df = pd.DataFrame(tar, columns=cn, index=rn)
print(df)
for i in range(data[:,0].shape[0]):
    #tar[int(data[i,0])-1, int(data[i,1])-1] = data[i,2]
    df.loc[data[i,0], data[i,1]] = int(data[i,2])
    print(data[i,2])
#df = pd.DataFrame(tar, columns=cn, index=rn)
print(df)

### output data 
df.to_csv('data/new_Sorted_500.txt', header=False, index=False, sep='\t', mode='a')