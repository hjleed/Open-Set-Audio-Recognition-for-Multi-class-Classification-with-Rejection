#!/usr/bin/env python2
#!/bin/python
# -*- coding: utf-8 -*-
# vim: set fileencoding=<classify.py> :
"""
This function to classify open set recognition

"""

__version_info__ = ('21', '07', '2019')
__version__ = '_'.join(__version_info__)
__author__ = 'Hitham Jleed'
print __doc__  
print 'version: ',__version__ 
print 'author: ',__author__ 

# importing 

# reset all variable
# like clear all in matlab
from IPython import get_ipython
get_ipython().magic('reset -sf') 




import numpy as np
import os
import random
import xlsxwriter

# detect the current working directory and print it
old_path = os.getcwd()  
os.chdir('C:/Users/hitham/Documents/Python_Scripts/libsvm/python')
from svmutil import *  # libSVM
os.chdir(old_path)  

CLASSES = ['knock', 'printer', 'keys', 'drawer', 'speech', 'keyboard']



#------- global definition -------


# --------classify-----------------------------
def peak_to_side_Ratio(P):
    P=list(P) 
    pmx=np.max(P)
    P.remove(max(P))
    PSR=(np.max(P)-np.min(P))/(np.std(P))
    return abs(pmx-PSR)

def most_common(lst):
    return max(set(lst), key=lst.count)


def classify(fname, dataSet,threshold):

    
    pred_list=[]
    prob_list=[]
    PSR_list=[]
    for item in dataSet[fname]: #frames in one audio file  
        predClazz, prob, pb = framepredict(item) 
        
        PSR=peak_to_side_Ratio(pb)
        
        if PSR > threshold:
            predClazz='unknown'        
        
        
        pred_list.append(predClazz)
        prob_list.append(prob)
        PSR_list.append(PSR)       
    
    return (pred_list,prob_list,PSR_list)
      
    
    ## EOF
def framepredict(item):
    maxProb = 0.0
    bestClass = ""
    pb = np.array([0]) # each frame, all classes prediction
    for clazz in CLASSES:
        mfile="models_30/model%s.txt"%(clazz)
        model = svm_load_model(mfile)
        output = svm_predict([0], [item], model, "-q -b 1")
        
        prob = output[2][0][0]
        pb = np.append(pb, prob)
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb, pb)
    ## EOF
    
   
        
        
def main():        
    
    thresholds=[1.95, 1.975, 2.0, 2.025, 2.05, 2.075, 2.10, 2.125, 2.15, 2.175, 2.20, 2.225, 2.25, 2.275, 2.3]
    for nm, thr in enumerate(thresholds):
        xlxfile="xlxs1/open_classify%d.xlsx"%(nm)
        row= 0
        workbook = xlsxwriter.Workbook(xlxfile)
        worksheet = workbook.add_worksheet()
        worksheet.write(row, 0, 'file')       # write_string()
        worksheet.write(row, 1, 'predicted')        # write_string()
        worksheet.write(row, 2, 'accuracy')           # write_number()
        worksheet.write(row, 3, 'PSR')            # write_number()
        
        #-----------------------------------------------------------------
        known_Data = np.load('feats/knownData.npy').item()
        known=[ 'knock15', 'printer19','printer17' ,'printer04', 'printer18','knock10',   
              'keys16' , 'keyboard19', 'printer16', 'printer20', 'printer09','printer01', 
              'printer15', 'printer11', 'printer03', 'knock07', 'printer02', 'knock11',   
              'printer07', 'knock18', 'printer05', 'knock08', 'knock09', 'knock20','keys12',    
              'keys09', 'keyboard18','keys10', 'printer12', 'speech04', 'drawer08','keys19',    
              'knock19', 'keys08', 'printer06', 'knock05', 'keyboard14', 'knock13', 'knock16',   
              'speech13','keys20','drawer04', 'knock21', 'keyboard11', 'speech18','keys18',    
              'speech07', 'keys07', 'keys02','knock03', 'keys13','keyboard15', 'speech12',  
              'speech16', 'keyboard17','printer08',     'speech05',      'knock06',     'keyboard12',
              'speech08', 'keys06', 'keyboard13', 'keyboard16',  'drawer10', 'keys03', 'keys01',   
              'keyboard02' , 'speech11', 'knock04', 'printer10', 'printer13' , 'keys14',  
              'printer14', 'speech15', 'knock17', 'speech14',   'drawer12',  'speech02' ,
              'speech17', 'keyboard20', 'keys11',  'keyboard05', 'drawer20', 'drawer13',    
              'keyboard07', 'keys15', 'speech19', 'speech20', 'knock02', 'speech09',  
              'speech10', 'drawer07', 'keyboard09',  'keyboard06', 'speech06', 'drawer22', 
              'speech01' , 'knock12' , 'knock01' , 'drawer16' , 'drawer03', 'speech03', 
              'drawer05', 'keys04' , 'keyboard04',  'keyboard01', 'drawer21', 'drawer09', 
              'drawer11', 'keyboard08', 'drawer14',  'keyboard21',  'drawer19'] 
    
       # known= known_Data.keys()
        print(len(known_Data.keys()))
    #------------------------
        unknown_Data = np.load('feats/unknownData.npy').item()    
        unknown=[ 'clearthroat10',    'pendrop03',   'doorslam09', 'pendrop20',  'doorslam17',   
        'doorslam06', 'mouse06', 'pendrop05', 'mouse09', 'laughter09', 'cough19', 'doorslam01',   
        'alert16',  'doorslam16', 'alert22', 'pageturn09', 'mouse10',  'pendrop16',  'cough11', 'mouse21', 'switch20',     
        'cough02',  'cough14',   'switch15', 'pageturn20', 'clearthroat20', 'mouse05',  'pendrop17',  'pageturn18',   
        'doorslam13', 'pendrop02', 'mouse07',  'pageturn01', 'switch19', 'doorslam19',  'pageturn04',  'pendrop07',  'mouse12',      
        'cough10', 'laughter14', 'doorslam15', 'switch13', 'mouse15', 'pendrop19', 'laughter03', 'clearthroat09',
        'pendrop14', 'pageturn11', 'cough16',  'cough07',  'phone21',  'alert06',  'switch10',  'alert08',      
        'pendrop08', 'cough15' , 'pendrop18',  'cough04', 'pageturn22', 'mouse02', 'laughter18',  'laughter21',   
        'doorslam08',  'doorslam18', 'pageturn07', 'clearthroat16', 'mouse01', 'laughter13',   
        'doorslam02', 'phone20', 'clearthroat15', 'cough01',  'switch06', 'laughter06',   
        'switch17', 'phone18', 'pageturn15', 'mouse18', 'pendrop04', 'switch08', 'laughter12', 'cough20',      
        'doorslam05', 'cough18', 'clearthroat02', 'pageturn10', 'laughter24', 'pageturn12' ,  
        'alert10', 'cough06',  'mouse20', 'cough05',  'doorslam11' , 'pageturn08', 'alert05', 'cough21',      
        'clearthroat18', 'laughter04', 'laughter07' ,'pageturn16', 'phone19', 'laughter23', 'laughter19',   
        'pendrop12', 'doorslam04',  'alert11',  'laughter01', 'alert17',  'pendrop09',  'phone11' , 'phone06',      
        'clearthroat06' ,'alert01', 'pendrop01', 'phone02', 'laughter08',  'mouse03' , 'pageturn14' ,  'clearthroat13'  ]
       # unknown= unknown_Data.keys()
        print(len(unknown_Data.keys()))
    
        #-----------------------------------------------------------------
    
        #select_n_known=15
        #select_n_unknown=15
    
        print("-----------------------")
       # print ("randomely selected %d files from known classes  \n" %(select_n_known) )
        print("-----------------------")
    
        
     #   known_list=random.sample(known,select_n_known)
        known_list=known
        for fname in known_list:
            print
            print ("tested file :  %s \n" %(fname) )
    
            pred_list,prob_list,PSR_list=classify(fname, known_Data,thr)
            print ("predicted class :  %s \n" %(most_common(pred_list)) )
    
            correct=pred_list.count(fname[:-2])
            print (" %d /%d of frames are correct" %(correct, len(pred_list) )) 
    
            for k, preName in enumerate(pred_list):
                #print(" %d: file ( %s )" %(k,preName))
                if preName == fname:
                    cc=1
                else:
                    cc=0
                    
                row +=1
                worksheet.write(row, 0, fname)                        # write_string()
                worksheet.write(row, 1, preName) 
                worksheet.write(row, 2, cc)           # write_number()
                worksheet.write(row, 3, PSR_list[k])            # write_number()
            
        print("-----------------------")
       # print ("randomely selected %d files from unknown classes  \n" %(select_n_unknown) )
    
    
        print("-----------------------")
        
    #    unknown_list=random.sample(unknown,select_n_unknown)
        unknown_list=unknown
        for fname in unknown_list:
            print
            print ("*** unknown to system ***")
            print ("tested file :  %s \n" %(fname) )
    
            pred_list,prob_list,PSR_list=classify(fname, unknown_Data, thr)    
            print ("predicted class :  %s \n" %(most_common(pred_list)) )
    
            correct=pred_list.count('unknown')
            print (" %d /%d of frames are correct" %(correct, len(pred_list) ))  
            
            for k, preName in enumerate(pred_list):
                #print(" %d: file ( %s )" %(k,preName))
                if preName == 'unknown' :
                    cc=1
                else:
                    cc=0
                    
                row +=1
                worksheet.write(row, 0, fname)                        # write_string()
                worksheet.write(row, 1, preName) 
                worksheet.write(row, 2, cc)           # write_number()
                worksheet.write(row, 3, PSR_list[k])            # write_number()
    
    
    
    
        workbook.close()  
    ## EOF
      
if __name__ == "__main__":
    main()        