### author: Zhichen Yan
### This file is used to remove :: symbol in MovieLens 1m dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

### open the original document
f = open ('Data/ratings.dat' , 'r')

### remove redundant :: symbol
data = np.array([line.split('::') for line in f])

df = pd.DataFrame(data)
print(df.dtypes)

### the timestamp will produce addtitional blank, so we set it to 0
df[3] = 0
df[0] = pd.to_numeric(df[0])
df[1] = pd.to_numeric(df[1])
df[2] = pd.to_numeric(df[2])
print(df.dtypes)
train, test = train_test_split(df, test_size=0.75)
print(test)
print(train.shape)
print(df[1].unique().shape[0])
### output data
df.to_csv('Data/rating.base', header=False, index=False, sep='\t', mode='a')
train.to_csv('Data/u2Data.train', header=False, index=False, sep='\t', mode='a')
test.to_csv('Data/u2Data.test', header=False, index=False, sep='\t', mode='a')
#train.to_csv('Data/u2.train', header=False, index=False, sep='\t', mode='a')
print("Save complete.")