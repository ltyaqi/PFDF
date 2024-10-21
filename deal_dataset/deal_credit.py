import numpy as np
import pandas as pd
coman = pd.read_csv('creditcard1.csv')
# print(coman.info())
coman.replace([np.inf, -np.inf], np.nan,inplace=True)
coman = coman.fillna(0)
def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        if (c == 'X2' or c =='X3' or c=='X4' or c =='X6' or c=='X7' or c=='X8' or c=='X9' or c=='X10' or c=='X11' or c=='Y'):
            newDataFrame[c] = df[c].tolist()
        else:
            d = df[c]
            MAX = d.max()
            MIN = d.min()
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame
data = regularit(coman)
outputpath = 'credit_all.csv'
data.to_csv(outputpath,sep=',',index=False,header=True) 