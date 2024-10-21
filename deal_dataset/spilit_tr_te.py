import csv
import json
import sys
import os
import pandas as pd
import numpy as np
import pickle
import random
data_len=30000
csv_file_path1='credit_all.csv'
csv_file_path2='credit_train.csv'
csv_file_path3='credit_test.csv'
csv_file_path4='credit_v.csv'
allfile=open(csv_file_path1,'r',encoding='UTF-8')
trainfile=open(csv_file_path2,'w',encoding='UTF-8')
testfile=open(csv_file_path3,'w',encoding='UTF-8')
vfile=open(csv_file_path4,'w',encoding='UTF-8')
train_idx = random.sample(range(1,data_len),int(data_len*0.7*0.7))
all_idx=[i+1 for i in range(data_len)] 
test_idx=random.sample(list(set(all_idx)-set(train_idx)),int(data_len*0.7*0.3))
v_idx=list(set(all_idx)-set(train_idx)-set(test_idx))
lines1=allfile.readlines()
trainfile.write(lines1[0])
testfile.write(lines1[0])
vfile.write(lines1[0])
for i in train_idx:
    trainfile.write(lines1[i]) 
trainfile.close()
for i in test_idx:
    testfile.write(lines1[i])
testfile.close() 
for i in v_idx:
    vfile.write(lines1[i])
vfile.close() 
allfile.close()