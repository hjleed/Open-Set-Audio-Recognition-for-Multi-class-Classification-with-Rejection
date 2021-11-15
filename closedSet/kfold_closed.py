# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:27:30 2019

@author: hitham
"""
# Load
import numpy as np
import os
import math
import sys
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0):
    sys.stdout.write("Sorry, requires Python 2.x, not Python 3.x\n")
    sys.exit(1)
#-------- global definition --------------------
Data = np.load('../data/DataDic1.npy').item()
CLASSES = Data.keys()
# libsvm constants
rbf =2  

# detect the current working directory and print it
old_path = os.getcwd()  
os.chdir('../libsvm-toolbox/python')
from svmutil import *  # libSVM
os.chdir(old_path)  

def setparameters (C=150, Gamma= .01):

    # parameters:
    param = svm_parameter("-s 0")
    param.probability = 1
    param.kernel_type = RBF
    #------------------
    param.C = C
    param.gamma = Gamma
    #print(param) 
    
    return (param)
  
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
    ## EOF
    return (labels, data)


# ============================Get features
def getLabeledDataVector(dataset, clazz, label):
    data = dataset[clazz]
    labels = [label] * len(data)
    output = zip(labels, data)          # python2
    #output = list(zip(labels, data))     # python3
    ## EOF
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
    ## EOF
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
    ## EOF
    return (bestClass, maxProb)


def plot_confusionMx(cm,title=None):

    if not title:
        title = 'confusion matrix'
    
    plt.close('all')
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

def Kfolddata(Data, k,f):
    
    param = setparameters ()
    tCount =0
    tCorrect =0
    confmatrix = np.zeros((len(CLASSES), len(CLASSES)))
    tstatic = np.zeros((len(CLASSES), 2))
    for current_fold in range(k):
        # start loop k-folding
        print ("----folding # %d -------------" % (current_fold+1))
        f.write("----folding # %d -------------" % (current_fold+1))
        trainingData = {}
        testData = {}
        for Clazz in CLASSES:
            # start loop classes
            X=Data[Clazz]
            #################################
            # size of fold
            fold_size = int(math.ceil(len(X)/k))
            # list of index
            ind=list(range(len(X)))
            # test index 
            test_ind=list(ind[current_fold*fold_size : ( current_fold + 1 ) * fold_size])
            test = X[test_ind]
            # the rest of index 
            train_ind = list([value for value in ind if value not in test_ind]) 
            train = X[train_ind]   
           # print (np.shape(test))
           # print (np.shape(train))
            ######################       
            trainingData[Clazz] = train        
            testData[Clazz] = test 
            # end loop classes
        models = getModels(trainingData,param)
        results= classify(models, testData)
        totalCount = 0
        totalCorrect = 0
        cm=[]
        static=[]
        for clazz in CLASSES:
            # start loop classes
            count, correct,pred_list = results[clazz]
            totalCount += count
            totalCorrect += correct
            print ("%s : %d , %d , %f \n" % (clazz, correct, count, (float(correct) / count))) 
            f.write("%s : %d , %d , %f\n" % (clazz, correct, count, (float(correct) / count))) 
            static.append([correct, count])
            row=[]
            for clazz in CLASSES:
                row.append(pred_list.count(clazz))
            cm.append(row)
            # end loop classes
        confmatrix=confmatrix+cm
        tstatic =tstatic + static
        tCount+=totalCount
        tCorrect +=totalCorrect 
        # end loop k-folding
        
    ## EOF

    return (confmatrix,tCount,tCorrect, tstatic)

   
def main():
    k=5 
    textfile="kfoldResult.txt"
    f = open(textfile, "w")
    confmatrix,t_samples,t_correct, static=Kfolddata(Data, k,f)

    print("----------- Overall --------\n")
    f.write("----------- Overall --------\n")
    for i in range(len(CLASSES)):
        print ("%s : %d , %d , %f\n" % (CLASSES[i], static[i][0], static[i][1], (float(static[i][0]) / static[i][1]))) 
        f.write("%s : %d , %d , %f\n" % (CLASSES[i], static[i][0], static[i][1], (float(static[i][0]) / static[i][1]))) 
    print ("%s %d %d %f\n" % ("Overall", t_correct, t_samples, (float(t_correct) / t_samples)))
    f.write("%s %d %d %f\n" % ("Overall", t_correct, t_samples, (float(t_correct) / t_samples)))
    plot_confusionMx(confmatrix,title='Confusion matrix')
    for i in range(len(confmatrix)):
        confmatrix[i]=confmatrix[i]/sum(confmatrix[i])*100
    confmatrix = np.round_(confmatrix, decimals = 2) # rounded to 2 significant digits
    plot_confusionMx(confmatrix,title='Normalized confusion matrix')
    f.close()
    ## EOF
    
if __name__ == "__main__":
    main()
    
 
 
