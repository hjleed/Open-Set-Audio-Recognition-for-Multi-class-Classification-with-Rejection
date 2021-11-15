#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
__author__ = 'Hitham Jleed'



import numpy as np
import os, glob
from scipy import signal
from sklearn.model_selection import KFold


# detect the current working directory and print it
old_path = os.getcwd()  
os.chdir('../libsvm-toolbox/python')
from svmutil import *  # libSVM
os.chdir(old_path)  

kfold = KFold(5, True, 1)
# Load
Data = np.load('../data/DataDic1.npy').item()
CLASSES = Data.keys()


def makefolding(Data):
    trainingData = {}
    testData = {}
    for Clazz in CLASSES:
        cl_data=Data[Clazz]
        #################################
        testSplit = int(.8 * len(cl_data))
        train = cl_data[:testSplit]
        test = cl_data[testSplit:]     
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
#    random.shuffle(labeledData)
    unzipped = [list(t) for t in zip(*labeledData)]
    labels, data = unzipped[0], unzipped[1]
    return (labels, data)


# ============================Get features
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
        for item in dataSet[trueClazz]:
            predClazz, prob = predict(models, item)
#            print ("%s,%s,%f" % (trueClazz, predClazz, prob))
            count += 1
#            if trueClazz != predClazz: print(item)
            if trueClazz == predClazz: correct += 1
        results[trueClazz] = (count, correct)
    return results

def predict(models, item):
    maxProb = 0.0
    bestClass = ""
    for clazz, model in models.iteritems():
        prob = predictSingle(model, item)
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb)

def predictSingle(model, item):
    
    #output = svm_predict([0], [item], model, "-q -b 1")
    output = svm_predict( [0], [item], model,'-q')

    prob = output[2][0][0]
    return prob
    
######## main    
def main():

    trainingData,testData=makefolding(Data)

        # parameters:
    param = svm_parameter("-s 0")
    # libsvm constants
    param.probability = 0
    param.kernel_type = RBF
    param.rbf =2
   # param.nr_fold= 5
   # param.cross_validation= True
    #C_raw = [0.01, 0.1, 1, 10, 100]
    C_raw = [1, 10, 50, 100, 150]
    gama_raw=[0.001, 0.01, 0.025, 0.5, 0.1]
    
    overall_accuracy = []
    hyperparams = []
    
    for c_m in C_raw:
        for g_m in gama_raw:

            #------------------
            param.C = c_m
            param.gamma = g_m
            #-------------------
            models = getModels(trainingData,param)
            
            results = classify(models, testData)
            
            totalCount = 0
            totalCorrect = 0
            
            for clazz in CLASSES:
                count, correct = results[clazz]
                totalCount += count
                totalCorrect += correct
            #    print ("%s : %d , %d , %f" % (clazz, correct, count, (float(correct) / count)))           
     #       print("----------------------------------------\n")
         #   print ("%s %d %d %f" % ("Overall", totalCorrect, totalCount, (float(totalCorrect) / totalCount)))
            overall_accuracy.append(float(totalCorrect) / totalCount)
            hyperparams.append((g_m, c_m))
    
    print ("---Optimizing Parameters ------ \n") 
    
    print ("overall_accuracy :")
    print(overall_accuracy)
    #f.write (overall_accuracy)
    print ("hyperparams: ",hyperparams)
    
    hyperparams_opt = hyperparams[overall_accuracy.index(max(overall_accuracy))]
    print("\n hyperparameters optimaization \n")
    print ("c= %d , gamma = % f \n" % (hyperparams_opt[1], hyperparams_opt[0]) )
    
    textfile="hyperparams_opt.txt"   
    f = open(textfile, "w") 
    f.write ("---Optimizied Parameters ------ \n" ) 
    f.write ("c= %d , gamma = % f \n" % (hyperparams_opt[1], hyperparams_opt[0]) )
    f.close()
    ## EOF
    
if __name__ == "__main__":
    main()
    