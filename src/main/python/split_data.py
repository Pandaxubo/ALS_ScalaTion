import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


f = open ('Data/ratings.dat' , 'r')

data = np.array([line.split('::') for line in f])

print(data.shape)
rn = np.array(data[:,0]) #row array
rn = np.unique(rn.astype(np.int)).astype(np.str)#row name
rns = np.unique(rn).size #row size
print(rns)
cn = np.array(data[:,1]) 
cn = np.unique(cn.astype(np.int)).astype(np.str)
cns = cn.size
print(cns)
tar = np.zeros([rns,cns])
print(tar)
df = pd.DataFrame(tar, columns=cn, index=rn)
print(df)
for i in range(data[:,0].shape[0]):
    #tar[int(data[i,0])-1, int(data[i,1])-1] = data[i,2]
    df.loc[data[i,0], data[i,1]] = int(data[i,2])
#df = pd.DataFrame(tar, columns=cn, index=rn)
print(df)

# df = pd.DataFrame(data)
# print(df.dtypes)
# df[3] = 0
# df[0] = pd.to_numeric(df[0])
# df[1] = pd.to_numeric(df[1])
# df[2] = pd.to_numeric(df[2])
# #print(df[3])
# print(df.dtypes)
# train, test = train_test_split(df, test_size=0.2)
# print(test)
# print(train.shape)
# df.to_csv('Data/rating.base', header=False, index=False, sep='\t', mode='a')
# train.to_csv('Data/u2Data.train', header=False, index=False, sep='\t', mode='a')
# test.to_csv('Data/u2Data.test', header=False, index=False, sep='\t', mode='a')
print("save complete")