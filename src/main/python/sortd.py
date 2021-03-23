import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

### open the original document
#f = open ('Data/u2Data.train' , 'r')
f = open ('Data/rating.txt' , 'r')
### remove redundant :: symbol
data = np.array([line.split('\t') for line in f])

df = pd.DataFrame(data)

df[0] = pd.to_numeric(df[0])
df[1] = pd.to_numeric(df[1])
sort = df.sort_values([0, 1], ascending = [True, True])
print(sort)

fr = open ('Data/a.txt' , 'r')
### remove redundant :: symbol
a = np.array([line.split(',') for line in fr])

dfa = pd.DataFrame(a)

print(dfa.iloc[6039,3818])