#!/usr/bin/env python2
#!/bin/python
# -*- coding: utf-8 -*-
# vim: set fileencoding=<opt_hyper_params.py> :
"""
-----------------------------------------
Optimizing parameters:   closedset/opt_hyper_params.py
The function is looking for search grid optimization looping for different values of C and Gamma.
 I used:     C = [0.001,  0.01,  0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 10, 20, 50, 100, 1000, 10000]  
      &  Gamma = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 5,  10, 20, 100, 1000, 10000]
Description: looping for all combination of C and Gamma, this function computes 5-fold classification.
 Each fold takes randomly 80% from each class as training and keeps 20% for testing via function named {makefolding(Data)}. 
 During the training , I used 1 vs all SVM.  6-classifiers were used. Each time one class is consider positive and the rest are negative. 
 In general, 6 models were created via function named {getModels(trainingData,param)}. During the testing.  
 The six classifiers compare the instance with their models by computing Platt posterior probability. 
 By voting, the classifier with higher probability won via function named {classify(models, dataSet)}. 
 Finally, the function compute hyper parameters for all values with all folding and decide which parameters are optimum for this classification.
The optimum parameters are printed hyperparams_opt.txt.

-------------------------------------------
"""
__author__ = 'Hitham Jleed'
print __doc__   # prints function's header comments

# reset all variable
from IPython import get_ipython
get_ipython().magic('reset -sf')  # similar to (clear all) in matlab

import numpy as np
import os
from sklearn.model_selection import KFold
import time
import pylab as pl
import xlsxwriter
from contextlib import contextmanager

@contextmanager
def timer(message):
    """
    Simple timing method. Logging should be used instead for large scale experiments.
    """
    print message
    start = time.time()
    yield
    stop = time.time()
    print "...elapsed time: {}".format(stop-start)


# detect the current working directory and print it
old_path = os.getcwd()  
#os.chdir('/home/hitham/Documents/libsvm/python')
os.chdir('C:/Users/hitham/Documents/Python_Scripts/libsvm/python')
from svmutil import *  # libSVM
os.chdir(old_path) 

kfold = KFold(5, True, 1)
# Load
#------- global definition -------
Data = np.load('feats1/knownData.npy').item()
CLASSES = ['knock', 'printer', 'keys', 'drawer', 'speech', 'keyboard']

def setparameters (C, Gamma):

    # parameters:
    param = svm_parameter("-s 0")
    param.probability = 1
    param.kernel_type = RBF
    #------------------
    param.C = C
    param.gamma = Gamma
    return (param)

def getLabeledDataVector(data, label):
    
    labels = [label] * len(data)
    output = zip(labels, data)          # python2
    #output = list(zip(labels, data))     # python3
    return output    
    ## EOF 
def getTrainingData(Data, selec):  
    labeledData=[]    
    for fname in Data.keys():
        newdata=Data[fname]
        
        if selec==fname[:-2]:  # positive class
            ld = getLabeledDataVector(newdata, 1)
            labeledData += ld
        else:                  # negative classess
            ld = getLabeledDataVector(newdata, -1)
            labeledData += ld
    np.random.shuffle(labeledData)
    unzipped = [list(t) for t in zip(*labeledData)]
    labels, data = unzipped[0], unzipped[1]
    return (labels, data)
    ## EOF

def getModels(Data,param):
    models = {}
    if not os.path.exists('models'):
        os.makedirs('models')
    for selec in CLASSES:
        labels, data = getTrainingData(Data, selec)
        prob = svm_problem(labels, data)
        m = svm_train(prob, param)
       # m_name="models/model%s.txt"%(selec)
       # svm_save_model(m_name, m)
        models[selec] = m
        print ("--training %s class is done--" %(selec))
    return models    


    
# classify


def classify(fname, dataSet,models):
    
    
    pred_list=[]
    prob_list=[]
    for item in dataSet[fname]: #frames in one audio file  
        predClazz, prob = framepredict(item, models)
       
        #-----------------    
        pred_list.append(predClazz)
        prob_list.append(prob)
    

    return pred_list,prob_list
        

def framepredict(item,models):
    maxProb = 0.0
    bestClass = ""
    pb = np.array([0]) # each frame, all classes prediction
    #for clazz in CLASSES:
        #mfile="models/model%s.txt"%(clazz)
        #model = svm_load_model(mfile)
    for clazz, model in models.iteritems():
        output = svm_predict([0], [item], model, "-q -b 1")
        
        prob = output[2][0][0]
        pb = np.append(pb, prob)
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb)
    ## EOF
def most_common(lst):
    return max(set(lst), key=lst.count)    
######## main    
def main():


    C_raw = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000, 10000]
    gama_raw=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 5, 10, 20, 100, 1000, 10000]
    
    row= 0
    workbook = xlsxwriter.Workbook('opt_parametrs.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(row, 0, 'C')       # write_string()
    worksheet.write(row, 1, 'gamma')        # write_string()
    worksheet.write(row, 2, 'Frame_acc')           # write_number()
    worksheet.write(row, 3, 'file_acc')            # write_number()
    
    overall_accuracy = []
    hyperparams = []
    E_overall_accuracy = []
    E_hyperparams = []
    files_list = Data.keys()
    for c_m in C_raw:
        for g_m in gama_raw:

            #------------------
            #param.C = c_m
            #param.gamma = g_m
            #-------------------
            
            param = setparameters (c_m, g_m)
            models = getModels(Data,param) 
            
            totalCount = 0
            totalCorrect = 0
            E_totalCount = 0
            E_totalCorrect = 0
            for fname in files_list:
                
                pred_list,prob_list=classify(fname, Data,models)
                #-----frames-----
                correct=pred_list.count(fname[:-2]) # number correct frames in the event
                count=len(pred_list)               # number of frames in each event
                totalCount += count
                totalCorrect += correct 
                #--------Events------
                E_totalCount += 1
                if most_common(pred_list) == fname[:-2]:
                    E_totalCorrect += 1  
            #---------------------------        
            overall_accuracy.append(float(totalCorrect) / totalCount)  #frames
            hyperparams.append((g_m, c_m))
            
            E_overall_accuracy.append(float(E_totalCorrect) / E_totalCount) 
            E_hyperparams.append((g_m, c_m))
            
            row +=1
            worksheet.write(row, 0, c_m)                        # write_string()
            worksheet.write(row, 1, g_m)       # write_string()
            worksheet.write(row, 2, float(totalCorrect) / totalCount)           # write_number()
            worksheet.write(row, 3, float(E_totalCorrect) / E_totalCount)            # write_number()
                                      
    workbook.close()  
    ## EOF
    
if __name__ == "__main__":
    main()
    
