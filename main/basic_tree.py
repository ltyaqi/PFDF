from math import log
import pandas as pd
import operator
import treePlotter
from collections import Counter
import random
import matplotlib.pyplot as plt
import re
import math
import numpy as np
from sklearn import metrics
weights={}
#label_all values(con is 0)
label_value = {}
#con_labels={'X1','X5','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23'}
con_labels={'age','trestbps','chol','thalach','oldpeak'}
#con_labels={'age','fnlwgt','education-num','hours-per-week'}
#con_labels={'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'}
#pre True:pre pruning;post True:post pruning
pre=False
post=False
#class label
data_class='target'
def read_dataset(filename,labels):
    #format:{label:[value]}
    data = pd.read_csv(filename)
    dataset=data.values.tolist()
    dataset_full={}
    n=len(label_value)
    for i in range(n):
        dataset_full[labels[i]]=[line[i] for line in dataset]
    return dataset_full
def kmeans(dataSet, k):
    #only can used for number
    #output:center value and all cluster value
    #print(len(dataSet))
    center_idx = random.sample(range(0,len(dataSet)),k)
    center=[dataSet[i] for i in center_idx]
    center.sort()
    while True:
        cluster=[]
        for i in range(k):
            cluster.append([])
        for i in dataSet:
            i=float(i)
            tmp=[]
            for t in range(k):
                tmp.append(abs(i-center[t]))
            n=tmp.index(min(tmp))
            cluster[n].append(i)
        centroids=[]
        for i in cluster:
            if len(i)!=0:
                centroids.append(sum(i)/len(i))
        if len(centroids)!=k:
            for i in range(k-len(centroids)):
                centroids.append(sum(centroids)/len(centroids))
        centroids.sort()
        if center==centroids:
            centroids.sort()
            return centroids, cluster
        center=centroids    
def con_value1(dataset):
    SSE=[]
    a=[]
    b=[]
    for i in range(5):
        t=0
        if len(dataset)<i+1:
            centroids,cluster=kmeans(dataset, len(dataset))
            a.append(centroids)
            b.append(cluster)
            length=len(centroids)
            for j in range(length):
                c=centroids[j]
                for m in cluster[j]:
                    t+=abs(m-c)**2
            SSE.append(t)
        else:
            centroids,cluster=kmeans(dataset, i+1)
            a.append(centroids)
            b.append(cluster)
            length=len(centroids)
            for j in range(length):
                c=centroids[j]
                for m in cluster[j]:
                    t+=abs(m-c)**2
            SSE.append(t)
    angle=[]
    for i in range(3):
        '''
        y1=SSE[i]-SSE[i+1]
        y2=SSE[i+2]-SSE[i+1]
        x1=-1
        x2=1
        t=x1*x2+y1*y2
        '''
        l3=((SSE[i]-SSE[i+2])**2+4)**0.5
        l1=(1+(SSE[i]-SSE[i+1])**2)**0.5
        l2=(1+(SSE[i+2]-SSE[i+1])**2)**0.5
        t1=((l1**2)+(l3**2)-(l2**2))/(2*l1*l3)
        t2=((l2**2)+(l3**2)-(l1**2))/(2*l2*l3)
        if t1>1 or t1< -1:
            a1=0
        else:
            a1=math.acos(t1)
        if t2>1 or t2 < -1:
            b1=0
        else:
            b1=math.acos(t2)
        angle.append(a1+b1)
    index=angle.index(max(angle))
    return a[1+index]
        
def con_value(dataset,n):
    #n is cluster num,output cluster center
    centroids,cluster=kmeans(dataset, n)
    return centroids
def cal_entropy(dataset):
    numEntries = len(dataset[data_class])
    labelCounts = {}
    for i in range(numEntries):
        currentlabel = dataset[data_class][i]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)  
    return Ent
def splitdataset(dataset, label, value):
    #according value to get dataset and return dataset dont include label_value(Discrete values)
    n=len(dataset[label]) 
    retdataset={} 
    d_index=[]
    for i in range(n):  
        if dataset[label][i] == value:
            d_index.append(i)
    for l_v in dataset:
        retdataset[l_v]=[dataset[l_v][j] for j in d_index]
    del retdataset[label]
    return retdataset
def cal_acc(test_output, label):
    #input:pre result ,true labels,output:acc
    assert len(test_output) == len(label)
    count = 0
    for index in range(len(test_output)):
        if test_output[index] == label[index]:
            count += 1

    return float(count / len(test_output))
def con_splitdataset(dataset,label,value1,value2):
    #(Continuous value)
    retdataset = {} 
    n=len(dataset[label]) 
    d_index=[]
    if value1==-1:
        for i in range(n):  #
            if float(dataset[label][i])<value2 or float(dataset[label][i]) == value2:
                d_index.append(i)
    if value2==-1:
        for i in range(n):  # 
            if float(dataset[label][i]) >value1 :
                d_index.append(i)
    if value1!=-1 and value2!=-1:
        for i in range(n):  # 
            if float(dataset[label][i]) > value1 and (float(dataset[label][i])<=value2 ) :
                d_index.append(i)
    for l_v in dataset:
        retdataset[l_v]=[dataset[l_v][j] for j in d_index]
    del retdataset[label]
    return retdataset
def splitdataset1(dataset, label, value):
    #keep label_values
    n=len(dataset[label]) 
    retdataset={} 
    d_index=[]
    for i in range(n):  
        if dataset[label][i] == value:
            d_index.append(i)
    for l_v in dataset:
        retdataset[l_v]=[dataset[l_v][j] for j in d_index]
    return retdataset

def con_splitdataset1(dataset,label,value1,value2):
    #keep label_values
    retdataset = {} 
    n=len(dataset[label]) 
    d_index=[]
    if value1==-1:
        for i in range(n):  #
            if float(dataset[label][i])<value2 or float(dataset[label][i]) == value2:
                d_index.append(i)
    if value2==-1:
        for i in range(n):  # 
            if float(dataset[label][i]) >value1 :
                d_index.append(i)
    if value1!=-1 and value2!=-1:
        for i in range(n):  # 
            if float(dataset[label][i]) > value1 and (float(dataset[label][i])<=value2 ) :
                d_index.append(i)
    for l_v in dataset:
        retdataset[l_v]=[dataset[l_v][j] for j in d_index]
    return retdataset
def ID3_chooseBestFeatureToSplit(dataset,labels):
    if len(dataset[data_class])==0:
        return -1,-1
    numFeatures = len(dataset) - 1
    baseEnt = cal_entropy(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(len(labels)-1):  
        newEnt = 0.0
        split_set=[]
        if labels[i] not in con_labels:
            featList = dataset[labels[i]]
            uniqueVals = label_value[labels[i]] # 
            for value in uniqueVals:  # 
                subdataset = splitdataset(dataset, labels[i], value)
                p = len(subdataset[data_class]) / float(len(dataset[data_class]))
                newEnt += p * cal_entropy(subdataset)
        else:
            tmpset=[float(i) for i in dataset[labels[i]]]
            #print(tmpset)
            v1=con_value1(tmpset)
            v1.append(-1)
            v1.insert(0,-1)
            for i1 in range(len(v1)-1):
                subdataset = con_splitdataset(dataset, labels[i], v1[i1],v1[i1+1])
                p = len(subdataset[data_class]) / float(len(dataset[data_class]))
                newEnt += p * cal_entropy(subdataset)
        infoGain = baseEnt - newEnt
        #######################################################################
        infoGain=infoGain*weights[labels[i]]
        ########################################################################
        if (infoGain > bestInfoGain or infoGain == bestInfoGain):
            bestInfoGain = infoGain  
            bestFeature = labels[i]
    return bestFeature,bestInfoGain
def local_chose_best(userdata,labels):
    best_label,best_ent=ID3_chooseBestFeatureToSplit(userdata,labels)
    return best_label,best_ent
def majorityCnt(classList):
    #vote get majority
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]
def ID3_createTree(dataset,labels,test_dataset,m):
    classList=[int(i) for i in dataset[data_class]]
    #print(classList)
    #if left labels num is m
    if len(classList)==0:
        return -1
    if len(labels)==m:
        return majorityCnt(classList)
    #if left flags are same
    if sum(classList) == len(classList) or sum(classList)==0:
        return classList[0]
    best_idx=[]
    best_ent=[]
    length=[]
    bestFeatLabel,b=local_chose_best(dataset,labels)
    uniqueVals = label_value[bestFeatLabel]
    #bestFeat=labels.index(bestFeatLabel)
    if pre:
        if len(test_dataset[data_class])==0:
                return majorityCnt(classList)
        else:
            ans = []
            for index in range(len(test_dataset[data_class])):
                ans.append(test_dataset[data_class][index])
            result_counter = Counter()
            for vec in dataset[data_class]:
                result_counter[vec] += 1
            leaf_output = result_counter.most_common(1)[0][0]
            root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset[data_class]), label=ans)
            outputs = []
            ans = []
            tmpset=[]
            if bestFeatLabel not in con_labels:
                for value in uniqueVals:
                    cut_testset = splitdataset(test_dataset, bestFeatLabel, value)
                    if len(cut_testset[data_class])==0:
                        continue
                    cut_dataset = splitdataset(dataset, bestFeatLabel, value)
                    for vec in cut_dataset[data_class]:
                        result_counter[vec] += 1
                    if len(cut_dataset[data_class])==0:
                        leaf_output1=leaf_output
                    else:
                        leaf_output1 = result_counter.most_common(1)[0][0]
                    for vec in cut_testset[data_class]:
                        ans.append(vec)
                    result_counter = Counter()
                    outputs += [leaf_output1] * len(cut_testset[data_class])
            else:
                tepset=[float(i) for i in dataset[bestFeatLabel]]
                value=con_value1(tmpset)
                v1=value[:]
                v1.append(-1)
                v1.insert(0,-1)
                temp=[]
                for i in range(len(v1)-1):
                    temp.append([v1[i],v1[i+1]])
                t1=0
                for t in temp:
                    cut_testset=con_splitdataset(test_dataset,bestFeatLabel,v1[t1],v1[t1+1])
                    if len(cut_testset[data_class])==0:
                        t1+=1
                        continue
                    cut_dataset=con_splitdataset(dataset,bestFeatLabel,v1[t1],v1[t1+1])
                    t1+=1
                    for vec in cut_dataset[data_class]:
                        result_counter[vec] += 1
                    if len(cut_dataset[data_class])==0:
                        leaf_output1=leaf_output
                    else:
                        leaf_output1 = result_counter.most_common(1)[0][0]
                    for vec in cut_testset[data_class]:
                        ans.append(vec)
                    result_counter = Counter()
                    outputs += [leaf_output1] * len(cut_testset[data_class])
            cut_acc = cal_acc(test_output=outputs, label=ans)
            if cut_acc <=root_acc:
                return leaf_output
    ID3Tree={bestFeatLabel: {}}
    labels.remove(bestFeatLabel)
    if bestFeatLabel not in con_labels:
        for value in uniqueVals:
            subLabels = labels[:]
            ID3Tree[bestFeatLabel][value] = ID3_createTree(
                splitdataset(dataset, bestFeatLabel, value),
                subLabels,
                splitdataset(test_dataset, bestFeatLabel, value),
                m)
            if ID3Tree[bestFeatLabel][value]==-1:
                t=[int(i) for i in dataset[data_class]]
                ID3Tree[bestFeatLabel][value]=majorityCnt(t)
    else:
        value=[]
        tmpset=[float(i) for i in dataset[bestFeatLabel]]
        value=con_value1(tmpset)
        v1=value[:]
        v1.append(-1)
        v1.insert(0,-1)
        t1=0
        temp=[]
        for i in range(len(v1)-1):
            temp.append([v1[i],v1[i+1]])
        
        for t in temp:
            t=str(t)
            subLabels = labels[:]
            ID3Tree[bestFeatLabel][t] = ID3_createTree(
                con_splitdataset(dataset,bestFeatLabel,v1[t1],v1[t1+1]),
                subLabels,
                con_splitdataset(test_dataset, bestFeatLabel,v1[t1],v1[t1+1]),
                m)
            t1+=1  
            if ID3Tree[bestFeatLabel][t]==-1:
                t2=[int(i) for i in dataset[data_class]]
                ID3Tree[bestFeatLabel][t]=majorityCnt(t2)     
    return ID3Tree
def classify(inputTree, featLabels, testVec):
    #according build tree to classify
    classLabel=testVec[data_class]
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if firstStr not in con_labels:
            if testVec[firstStr] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        else:
            find_lst=re.findall('-?\d+\.*\d*',key)
            v1=float(find_lst[0])
            v2=float(find_lst[1])
            if v1==-1:
                if float(testVec[firstStr]) <v2 or float(testVec[firstStr]) == v2:
                    if type(secondDict[key]).__name__ == 'dict':
                        classLabel = classify(secondDict[key], featLabels, testVec)
                    else:
                        classLabel = secondDict[key]
            if v2==-1:
                if float(testVec[firstStr]) >v1 or float(testVec[firstStr]) == v1:
                    if type(secondDict[key]).__name__ == 'dict':
                        classLabel = classify(secondDict[key], featLabels, testVec)
                    else:
                        classLabel = secondDict[key]
            if v1!=-1 and v2!=-1:
                if float(testVec[firstStr]) <v2 and float(testVec[firstStr]) >v1:
                    if type(secondDict[key]).__name__ == 'dict':
                        classLabel = classify(secondDict[key], featLabels, testVec)
                    else:
                        classLabel = secondDict[key]
    #print(classLabel)
    return classLabel
def classifytest(inputTree, featLabels, testDataSet):
    #testDataSet include many data and return class result according function:classify(one data)
    classLabelAll = []
    n=len(testDataSet[data_class])
    for i in range(n):
        testVec={}
        for label in featLabels:
            testVec[label]=testDataSet[label][i]
        t=classify(inputTree, featLabels, testVec)
        classLabelAll.append(int(t))
    return classLabelAll
def get_record(tree,l,record):
    firstStr = list(tree.keys())[0]
    record[firstStr][l]+=1
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            get_record(secondDict[key],l+1,record)
        else:
            return
    return record
if __name__ == '__main__':
#mpiexec -n num python filename.py(num=server+clients num)
    all_data="heart_UCI_all.csv"
    data = pd.read_csv(all_data)
    #get labels (include flag)
    labels=data.columns.tolist()
    #label-values if con:0
    for i in labels:
        if i in con_labels:
            label_value[i]=[0]
        else:
            label_value[i]=list(set(data[i]))
    
    train_file = "heart_UCI_train.csv"
    test_file = "heart_UCI_test.csv"
    v_file="heart_UCI_v.csv"
    '''
    train_file = "adult_train_all.csv"
    test_file = "adult_test_all.csv"
    v_file="v_dataset.csv"
    
    train_file = "credit_train.csv"
    test_file = "credit_test.csv"
    v_file="credit_v.csv"
    '''
    #alpha for weight calculate
    alpha=1.0
    dataset = read_dataset(train_file,labels)
    test_dataset=read_dataset(test_file,labels)
    v_dataset=read_dataset(v_file,labels)
    for i in range(len(labels)):
        weights.update({labels[i]:1.0})
    #m :max attribution one branch
    m=int((2*(len(labels)-1))/3)+1
    alpha=1.0
    trees=[]
    t2=[]
    for i in range(m):
        t2.append(0)
    for i in range(20):
        record={}
        for j in range(len(labels)):
            record.update({labels[j]:t2[:]})
        tree=ID3_createTree(dataset,labels[:],test_dataset,len(labels)-m)
        get_record(tree,0,record)
        '''
        for p in labels:
            t3=0 
            e=len(labels)
            for j in range(len(record[p])): 
                t3+=(5**(e-m-j))*(2**record[p][j])
            if t3!=0:
                weights[p]=weights[p]*((1.0/t3)*alpha)
        '''
        for p in labels:
            t3=0
            t3=2*sum(record[p])
        #t3=math.pow(2,sum(record[i]))
            if t3!=0:
                weights[p]=weights[p]*(1.0/t3)
        
        trees.append(tree)
        
        tmp=[]
        for tree in trees:
            tmp.append(classifytest(tree, labels,v_dataset))
        true_label= [int(i) for i in v_dataset[data_class]]
        t=[]
        for i in range(len(v_dataset[data_class])):
            t.append([data[i] for data in tmp])
        result=[]
        for line in t:
            result.append(int(max(line, key=line.count)))
        acc_final=cal_acc(result,true_label)
        print(acc_final)
        #print("auc",metrics.roc_auc_score(result,true_label))