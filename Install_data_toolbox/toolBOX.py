#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:00:40 2019

@author: hitham
"""
import os
import requests, zipfile, StringIO

def downloaddata(parentdir,datbase_link,libname):
    #if not os.path.exists(parentdir):
       # os.makedirs(parentdir)
    
    r = requests.get(datbase_link, stream=True)
    with zipfile.ZipFile(StringIO.StringIO(r.content)) as zf:
        zf.extractall(parentdir)
    for filename in os.listdir(parentdir):          
        src =parentdir+'/'+ filename 
        dst =parentdir+ '/' + libname 
        os.rename(src, dst)
       

def toolbox():
    
    parentdir='../toolbox'
    if not os.path.exists(parentdir):
        os.makedirs(parentdir)
    #--------------------------------
    libname='libSVM-3-24'   
    #libsvm-3.24     
    #Version 3.24 released on September 11, 2019. It conducts some minor fixes.     
    datbase_link='https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/libsvm-3.24.zip'
    downloaddata(parentdir,datbase_link,libname)  # from DCASE website

    #--------------------------------
    libname='libSVM-onevset'   
    #libSVM-onevset        
    #Extension of libSVM to support Open Set Recognitoin as described in "Toward Open Set Recognition", TPAMI July 2013    
    datbase_link='https://github.com/tboult/libSVM-onevset/archive/master.zip'
    downloaddata(parentdir,datbase_link,libname)  # from DCASE website
    
if __name__=="__main__":
    toolbox()
