#!/bin/bash
#install toolbox and database

echo "install DCASE-2013"
python download_data.py

echo "install LIBSVM toolbox"
python toolBOX.py
