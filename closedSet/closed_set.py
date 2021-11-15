#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""

# reset all variable
# like clear all in matlab
from IPython import get_ipython
get_ipython().magic('reset -sf') 

__author__ = 'Hitham Jleed'



import numpy as np
import os, glob
from scipy import signal

# detect the current working directory and print it
old_path = os.getcwd()  
os.chdir('../libsvm-toolbox/python')
from svmutil import *  # libSVM
os.chdir(old_path)  

#-------- global definition --------------------
# Load
Data = np.load('../data/DataDic1.npy').item()
CLASSES = Data.keys()

# libsvm constants
rbf =2

# parameters:
param = svm_parameter("-s 0")
param.probability = 1
param.kernel_type = RBF
nr_fold= 5
cross_validation= True
#------------------

 # got them from (opt_parameters.py)
                         # see (hyperparams_opt.txt)
param.C = 150
param.gamma = .025
# --------------------------------------------

def splitdata(Data):
    trainingData = {}
    testData = {}
    for Clazz in CLASSES:
        cl_data=Data[Clazz]
        np.random.shuffle(cl_data) 
        #################################
        testSplit = int(.2 * len(cl_data))
        print("class : %s  has %d instances  \n" % (Clazz, len(cl_data)))
        train = cl_data[testSplit:]  
        test  = cl_data[:testSplit]   
        ######################       
        trainingData[Clazz] = train        
        testData[Clazz] = test  

    return (trainingData, testData)
    
## train
def getModels(trainingData,param):
    models = {}
    for c in CLASSES:
        labels, data = getTrainingData(trainingData, c)
        prob = svm_problem(labels, data)
        m = svm_train(prob, param)
        models[c] = m
    return models

def getTrainingData(trainingData, clazz):    
    labeledData = getLabeledDataVector(trainingData, clazz, 1)
    negClasses = [c for c in CLASSES if not c == clazz]
    for c in negClasses:
        ld = getLabeledDataVector(trainingData, c, -1)
        labeledData += ld
    np.random.shuffle(labeledData)
    unzipped = [list(t) for t in zip(*labeledData)]
    labels, data = unzipped[0], unzipped[1]
    return (labels, data)


def getLabeledDataVector(dataset, clazz, label):
    data = dataset[clazz]
    labels = [label] * len(data)
    output = zip(labels, data)          # python2
    #output = list(zip(labels, data))     # python3
    return output    
    
# classify
def classify(models, dataSet):
    results = {}
    for trueClazz in CLASSES:
        count = 0
        correct = 0
        pred_list=[]
        for item in dataSet[trueClazz]:           
            predClazz, prob = framepredict(models, item)
            pred_list.append(predClazz)
            count += 1 
            if trueClazz == predClazz: correct += 1
        results[trueClazz] = (count, correct,pred_list)
    return results

def framepredict(models, item):
    maxProb = 0.0
    bestClass = ""
    pb = np.array([0])
    for clazz, model in models.iteritems():
        output = svm_predict([0], [item], model, "-q -b 1")
        prob = output[2][0][0]
        pb = np.append(pb, prob)
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb)

def plot_confusionMx(cm,title=None):
    
    if not title:
        title = 'confusion matrix'
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))  # size of figure
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j,i, str(cm[i][j]))
    plt.show()
    ## EOF
    
        
######## main  
    
def main():
    try:   
        trainingData,testData=splitdata(Data)

        #-------------------
        models = getModels(trainingData,param)
        
        results= classify(models, testData)
        
        totalCount = 0
        totalCorrect = 0
        confmatrix=[]

        for clazz in CLASSES:
            count, correct,pred_list = results[clazz]
            totalCount += count
            totalCorrect += correct
            print ("%s : %d , %d , %f" % (clazz, correct, count, (float(correct) / count))) 
            row=[]
            for clazz in CLASSES:
                row.append(pred_list.count(clazz))

            confmatrix.append(row)
           
            
        print("----------------------------------------\n")
        print ("%s %d %d %f" % ("Overall", totalCorrect, totalCount, (float(totalCorrect) / totalCount)))
        plot_confusionMx(confmatrix,title='Confusion matrix')


    except Exception as e:
        print (e)
        return 5
    
if __name__ == "__main__":
    main()

#