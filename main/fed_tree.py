from math import log
import pandas as pd
import operator
import treePlotter
from collections import Counter
import random
import matplotlib.pyplot as plt
from mpi4py import MPI
import re
import math
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn import metrics
#note: length flag==num_users
num_users=5
#len_rank=[0,1000,2000,5000,10000,20000]
#len_rank=[0,10,20,50,100,500]
len_rank=[0,2000,5000,10000,20000]
flag=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#######################################
#all labels' weight
weights={}
#label_all values(con is 0)
label_value = {}
#con_labels={'age','trestbps','chol','thalach','oldpeak'}
#con_labels={'age','fnlwgt','education-num','hours-per-week'}
#con_labels={'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'}
con_labels={'X1','X5','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23'}
#pre True:pre pruning;post True:post pruning
pre=False
post=False
d_p=False
#keep comunication synchronous
t5=0
tree_num=5
tree_deep=1.0/3
#class label
data_class='Y'
#epsilon
epsilon_sum=2.5/tree_num
exp_epsilon=0.5
C=1
def exp_machanism(sensitivety,epsilon,x):
    ele_num=len(x)
    p=[math.exp(float(epsilon*(i/1000))/sensitivety) for i in x]
    p_sum=sum(p)
    p1=[i/p_sum for i in p[:]]
    p1_length=len(p1)
    p2=[sum(p1[:i+1]) for i in range(p1_length)]
    random_value=random.random()
    p2.insert(0,0)
    for i in range(ele_num):
        if random_value>=p2[i] and random_value<p2[i+1]:
            return i         
def laplace_noise(sensitivety,epsilon):
    n_value = np.random.laplace(0, sensitivety/epsilon, 1)
    return n_value
# output: data+laplace noise
def laplace_mech(data, sensitivety, epsilon):
    data+= laplace_noise(sensitivety,epsilon)
    return data
def dist_datasets_iid(dataset, num_users,labels):
    user_data=[]
    n=len(dataset[labels[0]])
    num_items = int(n/num_users)
    all_idx=set([i for i in range(n)])
    for i in range(num_users):
        data_idx = random.sample(list(all_idx),num_items)
        tem_dic={}
        for label in labels:
            tem_dic[label]=[dataset[label][j] for j in data_idx]
        user_data.append(tem_dic)
        all_idx=all_idx-set(data_idx)
    return user_data
def dist_datasets_noniid(dataset,labels,rate1,rate2):
    user_data=[]
    n=len(dataset[labels[0]])
    num_items = int(n*rate1)
    all_idx=set([i for i in range(n)])
    data_idx = random.sample(list(all_idx),num_items)
    tem_dic={}
    tem_dic1={}
    tem_dic2={}
    for label in labels:
        tem_dic[label]=[dataset[label][j] for j in data_idx]
    user_data.append(tem_dic)
    all_idx=list(all_idx-set(data_idx))
    data_idx1 = random.sample(all_idx,int(n*rate2))
    for label in labels:
        tem_dic1[label]=[dataset[label][j] for j in data_idx1]
    user_data.append(tem_dic1)
    all_idx=list(set(all_idx)-set(data_idx1))
    for label in labels:
        tem_dic2[label]=[dataset[label][j] for j in all_idx]
    user_data.append(tem_dic2)
    return user_data
def dist_dataset_iid1(dataset,num_users,labels):
    rate=0.7
    test_data=[]
    train_data=[]
    n=len(dataset[labels[0]])
    train_num=int(rate*n)
    all_idx=set([i for i in range(n)])
    for i in range(num_users):
        data_idx = random.sample(list(all_idx),train_num)
        tem_dic={}
        for label in labels:
            tem_dic[label]=[dataset[label][j] for j in data_idx]
        train_data.append(tem_dic)
        test_idx=list(all_idx-set(data_idx))
        t_d={}
        for label in labels:
            t_d[label]=[dataset[label][j] for j in test_idx]
        test_data.append(t_d)
    return train_data,test_data
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
def con_random_value(dataset):
    split_num=2
    if len(list(set(dataset)))==1:
        return [dataset[0]]
    b = random.sample(list(set(dataset)), split_num) 
    return b
def con_value1(dataset):
    #Elbow method (SSE(sum of the squared errors）) dont need to point special cluster num ,but need to set range (this 1-5)
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
            if float(dataset[label][i]) >value1:
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
def majorityCnt(classList):
    #vote get majority
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]
def ID3_chooseBestFeatureToSplit(dataset,labels,gain_epsilon):
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
            v1=con_random_value(tmpset)
            v1.append(-1)
            v1.insert(0,-1)
            for i1 in range(len(v1)-1):
                subdataset = con_splitdataset(dataset, labels[i], v1[i1],v1[i1+1])
                p = len(subdataset[data_class]) / float(len(dataset[data_class]))
                newEnt += p * cal_entropy(subdataset)
        infoGain = (baseEnt - newEnt)*weights[labels[i]]
        if d_p:
            infoGain=laplace_mech(infoGain, 1, gain_epsilon)
            if infoGain<0.0:
                infoGain=0.0
        if (infoGain > bestInfoGain or infoGain == bestInfoGain):
            bestInfoGain = infoGain  
            bestFeature = labels[i]
    return bestFeature,bestInfoGain
def local_chose_best(userdata,labels,gain_epsilon):
    best_label,best_ent=ID3_chooseBestFeatureToSplit(userdata,labels,gain_epsilon)
    return best_label,best_ent
def server_chose_best(best_label,best_ent,length):
    best_idx_set=set(best_label)
    max_idx=best_label[0]
    max_ent=0
    for idx in best_idx_set:
        sum_length=0
        tmp=[]
        for i in range(len(best_label)):
            if idx==best_label[i]:
                sum_length=sum_length+length[i]
                tmp.append(i)
        if sum_length!=0:
            t=0
            for i in tmp:
                t=t+(float(length[i])/sum_length)*best_ent[i]
            #t=t*weights[idx]
            if t>max_ent:
                max_ent=t
                max_idx=idx
    return max_idx
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
def fake_tree(user,value):
    #In order to keep the communication synchronized, let the users who can already end (a user is already a leaf node, but other users can continue to divide) continue to communicate with the server side (not needed in reality, this is required to use MPI here, otherwise the transmission information will be chaotic)
    comm.send(-1, dest=0, tag=user+15)
    f_r=comm.recv(source=0, tag=user+3)
    if f_r==-2:
        return value
    if pre:
        if f_r in con_labels:
            comm.send([0,0], dest=0, tag=user+10)     
            value1=comm.recv(source=0, tag=user+20)
        comm.send(-1, dest=0, tag=user+40)
        jud=comm.recv(source=0, tag=user+45)
        if jud==0:
            return value
    ID3Tree={f_r: {}}
    if f_r not in con_labels:
        uniqueVals = label_value[f_r]
        for value1 in uniqueVals:
            ID3Tree[f_r][value1] = fake_tree(user,value)
    else:
        tt=[]
        comm.send(tt, dest=0, tag=user+10) 
        value1=comm.recv(source=0, tag=user+20)
        v1=value1[:]
        v1.append(-1)
        v1.insert(0,-1)
        t1=0
        temp=[]
        for i in range(len(v1)-1):
            temp.append([v1[i],v1[i+1]])
        for t in temp:
            t=str(t)
            ID3Tree[f_r][t] = fake_tree(user,value)
    return ID3Tree
    
def ID3_createTree(user,dataset,labels,test_dataset,m,gain_epsilon,centroids_epsilon):
    #，input:user_id、、testdata for pruning、m=from root to leaf max length
    classList=[int(i) for i in dataset[data_class]]
    #if dataset length=0 return -1
    if len(dataset[data_class])==0:
        return -1
    #if left labels num is m
    if len(labels)==m:
        return fake_tree(user,majorityCnt(classList))
    #if left flags are same
    if sum(classList) == len(classList) or sum(classList)==0:
        return fake_tree(user,classList[0])
    best_idx=[]
    best_ent=[]
    a,b=local_chose_best(dataset,labels,gain_epsilon)
    #print(a,b)
    #b=laplace_mech(b, 1, gain_epsilon)
    if a==-1:
        print([a,b],labels)
    comm.send([a,b], dest=0, tag=user+15)
    #get best feature from server
    bestFeatLabel=comm.recv(source=0, tag=user+3)
    #print(user,bestFeatLabel)
    uniqueVals = label_value[bestFeatLabel]
    #bestFeat=labels.index(bestFeatLabel)
    if pre:
        if len(test_dataset[data_class])==0:
            if bestFeatLabel in con_labels:
                comm.send([0,0], dest=0, tag=user+10)     
                value=comm.recv(source=0, tag=user+20)
            comm.send(-1, dest=0, tag=user+40)#notestdatanone
            jud=comm.recv(source=0, tag=user+45)
            if jud==0:
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
                tt=con_random_value(tmpset)
                comm.send(tt,dest=0,tag=user+10)
                value=comm.recv(source=0, tag=user+20)
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
            #print(cut_acc,root_acc)
            if cut_acc <root_acc:
                comm.send(0, dest=0, tag=user+40)       
            else:
                comm.send(1, dest=0, tag=user+40)
            jud=comm.recv(source=0, tag=user+45)
            if jud==0:
                return leaf_output
    #weights[bestFeatLabel][0]=weights[bestFeatLabel][0]+1
    #dict tree
    ID3Tree={bestFeatLabel: {}}
    labels.remove(bestFeatLabel)
    if bestFeatLabel not in con_labels:
        for value in uniqueVals:
            subLabels = labels[:]
            ID3Tree[bestFeatLabel][value] = ID3_createTree(
                user,
                splitdataset(dataset, bestFeatLabel, value),
                subLabels,
                splitdataset(test_dataset, bestFeatLabel, value),
                m,gain_epsilon,centroids_epsilon)
            if ID3Tree[bestFeatLabel][value]==-1:
                t=[int(i) for i in dataset[data_class]]
                ID3Tree[bestFeatLabel][value]=fake_tree(user,majorityCnt(t))
    else:
        value=[]
        tmpset=[float(i) for i in dataset[bestFeatLabel]]
        tt=con_random_value(tmpset)
        if d_p:
            tt=[abs(laplace_mech(it, C, centroids_epsilon)) for it in tt[:]]
        comm.send(tt, dest=0, tag=user+10) 
        value=comm.recv(source=0, tag=user+20)
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
                user,
                con_splitdataset(dataset,bestFeatLabel,v1[t1],v1[t1+1]),
                subLabels,
                con_splitdataset(test_dataset, bestFeatLabel,v1[t1],v1[t1+1]),
                m,gain_epsilon,centroids_epsilon)
            t1+=1
            if ID3Tree[bestFeatLabel][t]==-1:
                t2=[int(i) for i in dataset[data_class]]
                ID3Tree[bestFeatLabel][t]=fake_tree(user,majorityCnt(t2))         
    return ID3Tree

def server_tree(deep,user_data_rank):
#one tree 
    a=0
    best_label=[]
    best_ent=[]
    for i in range(num_users):
        if flag[i]==1:
            #receive clients ' best attribution 
            t=comm.recv(source=i+1, tag=i+15)
            if t==-1:
                a=a+1
            else:
                best_label.append(t[0])
                best_ent.append(t[1])
    #when all clients are all leaf,end this branch    
    if a==num_users:
        for i in range(num_users):
            comm.send(-2, dest=i+1, tag = i+3)
        return -1
    #sever choose the best attribution and send it to clients
    choose_label=server_chose_best(best_label,best_ent,user_data_rank)
    for i in range(num_users):
        comm.send(choose_label, dest=i+1, tag = i+3)
    #print(choose_label)
    if pre:
        #pre-pruning
        if choose_label in con_labels:
            ttt=[]
            for i in range(num_users):
                tt=comm.recv(source=i+1, tag=i+10)
                for j in tt:
                    ttt.append(j)
            tttt=con_random_value(ttt)
            for i in range(num_users):
                comm.send(tttt, dest=i+1, tag =i+20)
        jud=[]
        for i in range(num_users):
            t4=comm.recv(source=i+1, tag=i+40)
            if t4!=-1:
                jud.append(t4)
        if jud==[]:
            for i in range(num_users):
                comm.send(1, dest=i+1, tag = i+45)
        else:
            if sum(jud)<len(jud)/2 and deep!=0:
                for i in range(num_users):
                    comm.send(0, dest=i+1, tag = i+45)
                return -1
            else:
                for i in range(num_users):
                    comm.send(1, dest=i+1, tag = i+45)
    ID3Tree={choose_label: {}}
    deep=deep+1
    if choose_label in con_labels:
        ttt=[]
        for i in range(num_users):
            tt=comm.recv(source=i+1, tag=i+10)
            for j in tt:
                ttt.append(j)
        tttt=con_random_value(ttt)
        for i in range(num_users):
            comm.send(tttt, dest=i+1, tag =i+20)
        v1=tttt[:]
        v1.append(-1)
        v1.insert(0,-1)
        #print(v1)
        temp=[]
        for i in range(len(v1)-1):
            temp.append([v1[i],v1[i+1]])
        for i in temp:
            i=str(i)
            ID3Tree[choose_label][i] = server_tree(deep,user_data_rank)
    else:
        uniqueVals = label_value[choose_label]
        for value in uniqueVals:
            ID3Tree[choose_label][value] = server_tree(deep,user_data_rank)
        
    return ID3Tree
            
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
def cal_acc(test_output, label):
    #input:pre result ,true labels,output:acc
    assert len(test_output) == len(label)
    count = 0
    for index in range(len(test_output)):
        if test_output[index] == label[index]:
            count += 1

    return float(count / len(test_output))
def get_leaf(inputTree,leaf):
    #get leaf values from a tree
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            classLabel = get_leaf(secondDict[key],leaf)
        else:
            #print(secondDict)
            leaf.append(secondDict[key])
    return leaf
def get_alltree(tree,global_leaf):
    #combine tree structure and leaf
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            global_leaf,new_tree=get_alltree(secondDict[key],global_leaf)
        else:
            secondDict[key]=global_leaf[0]
            global_leaf=global_leaf[1:]
    new_tree=tree
    return global_leaf,new_tree
def server_post_pruning(inputTree):
    firstStr=list(inputTree.keys())[0]
    idx=labels.index(firstStr)
    secondDict=inputTree[firstStr]
    pruningTree={firstStr:{}} 
    contrastTree={firstStr:{}} 
    for key in secondDict.keys():
        contrastTree[firstStr][key]=-1 
        if type(secondDict[key]).__name__=='dict':
                pruningTree[firstStr][key]=server_post_pruning(secondDict[key])
        else:
            pruningTree[firstStr][key]=secondDict[key]
    jud=[]
    for i in range(num_users):
        #receive clients whether pruning or not
        t4=comm.recv(source=i+1, tag=i+60)
        if t4!=-1:
            jud.append(t4)
    #according clients to make a choice and send result to clients
    if jud==[]:
        for i in range(num_users):
            comm.send(1, dest=i+1, tag = i+65)
        return pruningTree
    else:
        if sum(jud)<len(jud)/2:
            for i in range(num_users):
                comm.send(0, dest=i+1, tag = i+65)
            return -1
        else:
            for i in range(num_users):
                comm.send(1, dest=i+1, tag = i+65)
            return pruningTree
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
def server_main(m,labels,user_data_rank):
    t2=[]
    for i in range(m):
        t2.append(0)
    #record information (will be used for weight)
    record={}
    for i in range(len(labels)):
        record.update({labels[i]:t2[:]})
    #server tree (leaf is -1)
    tree=server_tree(0,user_data_rank)
    if post:
    #server post
        tree=server_post_pruning(tree)
    get_record(tree,0,record)
    #according decision to caculate attributions' new weight 
    for i in labels:
        t3=0 
        
        t3=2*sum(record[i])
        #t3=math.pow(2,sum(record[i]))
        if t3!=0:
            weights[i]=weights[i]*(1.0/t3)
        '''
        e=len(labels)
        for j in range(len(record[i])): 
            t3+=(5**(e-m-j))*(2**record[i][j])
        if t3!=0:
            weights[i]=weights[i]*((1.0/t3)*alpha)
         ''' 
    #treePlotter.ID3_Tree(tree)
    #according clients' leaf to get global tree leaf
    leaf=[]
    acc=[]
    length=user_data_rank
    for i in range(num_users):
        #t6 include leaf/acc/dataset len
        t6=comm.recv(source=i+1, tag=i+26)
        #print(i,t6)
        leaf.append(t6[0])
        acc.append(t6[1])
    l=len(leaf[0])
    global_label=[0 for i in range(l)]
    length1=0
    for i in range(l):
        t=[j[i] for j in leaf]
        t1=set(t)
        max1=0
        for m1 in t1:
            tmp=0
            for mm in range(len(t)):
                if t[mm]==m1:
                    tmp=tmp+acc[mm]*length[mm]
            if tmp>max1:
                global_label[i]=m1
                max1=tmp
    #print(global_label)
    #get final global decision tree
    tn,global_tree=get_alltree(tree,global_label)
    #treePlotter.ID3_Tree(global_tree)
    return global_tree
def valPrecision(thisTree,valdata,labels):
    #get acc
    truelabels=[int(i) for i in valdata[data_class]]
    result=classifytest(thisTree, labels[:],valdata)
    acc=cal_acc(result, truelabels)
    return acc 
def createPostpruningTree(user,inputTree,dataset,test_dataset,labels):
    #client post
    firstStr=list(inputTree.keys())[0] 
    pruningTree={firstStr:{}} 
    contrastTree={firstStr:{}} 
    idx=labels.index(firstStr)
    secondDict=inputTree[firstStr]
    typedfdata=0
    if len(dataset[data_class])!=0:
        classList=[int(i) for i in dataset[data_class]]
        typedfdata=majorityCnt(classList) 
    else:
        a=[]
        typedfdata=get_leaf(inputTree,a)[0]
    if len(test_dataset[data_class])==0:
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                pruningTree[firstStr][key]=createPostpruningTree(user,secondDict[key],dataset,test_dataset,labels)
            else:
                pruningTree[firstStr][key]=secondDict[key]
        comm.send(-1, dest=0, tag = user+60)
    else: 
        for key in secondDict.keys():
            contrastTree[firstStr][key]=typedfdata 
            if type(secondDict[key]).__name__=='dict':
                if firstStr not in con_labels:
                    pruningTree[firstStr][key]=createPostpruningTree(user,secondDict[key],splitdataset1(dataset,firstStr,key),splitdataset1(test_dataset,firstStr,key),labels)
                else:
                    #use regular expression to get number in string
                    find_lst=re.findall('-?\d+\.*\d*',key)
                    v1=float(find_lst[0])
                    v2=float(find_lst[1])
                    pruningTree[firstStr][key]=createPostpruningTree(user,secondDict[key],con_splitdataset1(dataset,firstStr,v1,v2),con_splitdataset1(test_dataset,firstStr,v1,v2),labels)
            else:
                pruningTree[firstStr][key]=secondDict[key]
        precisionContrast=valPrecision(contrastTree,test_dataset,labels[:])
        precisionPruning=valPrecision(pruningTree,test_dataset,labels[:])
        #print(precisionContrast,precisionPruning)
        if precisionContrast>precisionPruning:
        #if not pruning acc>pruning acc ,send 0
            comm.send(0, dest=0, tag = user+60)
        else:
        #if <= send 1
            comm.send(1, dest=0, tag = user+60)
    jud=comm.recv(source=0, tag=user+65)
    #print(pruningTree)
    if jud==1:
    #1 pruning
        return pruningTree
    else:
    #0 not pruning
        return typedfdata
def length_score(length):
    t=len(len_rank)
    median=[]
    for i in range(t-1):
        median.append(len_rank[i+1]-len_rank[i])
    scores=[]
    for i in length:
        #scores.append([abs(i-j) for j in median])
        scores.append([float(1)/abs(i-j) for j in median])
    return scores

if __name__ == '__main__':
#mpiexec -n num python filename.py(num=server+clients num)
    #all_data="adult_train_all.csv"
    all_data="credit_all.csv"
    #all_data="car_evaluation.csv"
    data = pd.read_csv(all_data)
    #get labels (include flag)
    labels=data.columns.tolist()
    class_value=list(set(data[data_class]))
    #label-values if con:0
    for i in labels:
        if i in con_labels:
            label_value[i]=[0]
        else:
            tm=[j for j in data[i]]
            tmt=list(set(tm))
            tmt.sort(key=tm.index)
            label_value[i]=tmt[:]
    '''       
    train_file = "adult_train_all.csv"
    test_file = "adult_test_all.csv"
    v_file="v_dataset.csv"
    train_file = "heart_UCI_train.csv"
    test_file = "heart_UCI_test.csv"
    v_file="heart_UCI_v.csv"
    '''
    train_file = "credit_train.csv"
    test_file = "credit_test.csv"
    v_file="credit_v.csv"
    
    #alpha for weight calculate
    data1=read_dataset(all_data,labels)
    dataset = read_dataset(train_file,labels)
    test_dataset=read_dataset(test_file,labels)
    v_dataset=read_dataset(test_file,labels)
    #m :max attribution one branch
    m=int((len(labels)-1)*tree_deep)+1
    for i in range(len(labels)):
        weights.update({labels[i]:1.0})
    #print("max_deep:",m)
    #random divide dataset to users
    #user_data=dist_datasets_iid(dataset, num_users,labels)
    #user_data,user_testdata=dist_dataset_iid1(data1, num_users,labels)
    #user_data_length=[len(i[data_class]) for i in user_data]
    #user_testdata=dist_datasets_iid(test_dataset, num_users,labels)
    #user_testdata=dist_datasets_iid1(test_dataset, num_users,labels)
    '''
    dataset_length=[len(user_data[i][data_class])+len(user_testdata[i][data_class]) for i in range(num_users)]
    data_len_rank=[0 for i in range(num_users)]
    user_id=0
    scores=length_score(dataset_length)
    data_len_rank=[]
    for i in scores:
        if d_p:
            data_len_rank.append(exp_machanism(1,exp_epsilon,i)+2)
        else:
            data_len_rank.append(i.index(min(i))+1)
            '''
    #MPI config
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nworkers = comm.Get_size()-1
    workers = list(range(1,nworkers+1))   
    #rank=0:server，rank>0:clients Run a local program and block receiving and sending information
    if rank!=0:
        train_data=comm.recv(source=0, tag=rank+56)
        test_data=comm.recv(source=0, tag=rank+57)
        dataset_length=[len(train_data[data_class])+len(test_data[data_class])]
        data_len_rank=0
        scores=length_score(dataset_length)
        for i in scores:
            if d_p:
                data_len_rank=exp_machanism(1,exp_epsilon,i)+2
            else:
                data_len_rank=i.index(min(i))+1
        print("user",rank)
        ##############################################################################################
        comm.send(data_len_rank,dest=0,tag=rank+99)
        #xiaorongshiyan
        #comm.send(2,dest=0,tag=rank+99)
        #generate m local trees
        for i in range(tree_num):
            tree=ID3_createTree(rank-1,train_data,labels[:],test_data,len(labels)-m,epsilon_sum/min((m*2+1),len(con_labels)+m+1),epsilon_sum/min((m*2+1),len(con_labels)+m+1))
            #tree=ID3_createTree(rank-1,train_data,labels[:],test_data,len(labels)-m,epsilon_sum/(m+1),0)
            if post:
                tree=createPostpruningTree(rank-1,tree,train_data,test_data,labels[:])
            result=classifytest(tree, labels[:], test_data)
            true_label=[int(j) for j in test_data[data_class]]
            acc=cal_acc(result, true_label)
            leaf=[]
            leaf1=get_leaf(tree,leaf)
            if d_p:
                acc=laplace_mech(acc, 1, epsilon_sum/min((m*2+1),len(con_labels)+m+1))
            #send final leaf /acc/length of dataset to server
            comm.send([leaf1,acc], dest=0, tag = rank+25)
            tree_global=comm.recv(source=0, tag=rank+43)
            record={}
            t2=[]
            for i3 in range(m):
                t2.append(0)
            for i1 in labels[:-1]:
                record.update({i1:t2[:]})
            get_record(tree_global,0,record)
            #according decision to caculate attributions' new weight 
            '''
            alpha=1.0
            for i2 in labels[:-1]:
                t3=0 
                t3=2*sum(record[i2])
                if t3!=0:
                    weights[i2]=weights[i2]*(1.0/t3)
                    '''

    if rank==0:
        #user_data=dist_datasets_noniid(dataset,labels,1.0/10,1.0/10)
        #user_testdata=dist_datasets_noniid(test_dataset,labels,1.0/10,1.0/10)
        user_data=dist_datasets_iid(dataset,num_users,labels)
        user_testdata=dist_datasets_iid(test_dataset,num_users,labels)
        for j in range(num_users):
            comm.send(user_data[j],dest=j+1,tag=j+57)
            comm.send(user_testdata[j],dest=j+1,tag=j+58)
    #server
        print(rank)
        
        user_rank=[]
        for user_id in range(num_users):
            user_rank.append(comm.recv(source=user_id+1, tag=user_id+100))
        #print("1")
        trees=[]
        #m trees
        for i in range(tree_num):
            #get global tree
            tree=server_main(m,labels[:-1],user_rank)
            for j in range(num_users):
                comm.send(tree,dest=j+1,tag=j+44)
            #draw
            #treePlotter.ID3_Tree(tree)
            trees.append(tree)
        #vote and get final result:result
            tmp=[]
            for tree1 in trees:
                tmp.append(classifytest(tree1, labels,v_dataset))
            true_label= [int(i) for i in v_dataset[data_class]]
            t=[]
            for i1 in range(len(v_dataset[data_class])):
                t.append([data[i1] for data in tmp])
            result=[]
            for line in t:
                result.append(int(max(line, key=line.count)))
            acc_final=cal_acc(result,true_label)
            f1_sc=f1_score(true_label, result, average='weighted')
            print(i+1,"Accuracy:",acc_final,", f1_score:",f1_sc)
            
    

            
                
    
