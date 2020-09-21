# -*- coding: utf-8 -*-
"""

IEEE AASP CASA Challenge - Training Dataset for Event Detection Task (subtasks OL, OS)


IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events (http://www.elec.qmul.ac.uk/digitalmusic/sceneseventschallenge/). Training Dataset for Event Detection Task, subtasks 1 - OL and 2 - OS. Dataset developers: Dimitrios Giannoulis, Emmanouil Benetos, Dan Stowell, and Mark Plumbley (Queen Mary University of London) The training dataset for the OL and OS subtasks consists of 3 files: singlesounds_stereo.zip, singlesounds_bformat.zip, singlesounds_annotation.zip. Please refer to 'README.txt' inside the .zip files for more information.


https://c4dm.eecs.qmul.ac.uk/rdr/handle/123456789/28
"""

import os


def downloaddata(datbase_link, dest_dir):
    if not os.path.exists(dest_dir):
        import requests, zipfile, StringIO
        r = requests.get(datbase_link, stream=True)
        with zipfile.ZipFile(StringIO.StringIO(r.content)) as zf:
            zf.extractall(dest_dir)
    else:
        print ('%s File is already exist' %(dest_dir))     
        
if __name__ == "__main__":
   
    datbase_link='https://c4dm.eecs.qmul.ac.uk/rdr/bitstream/handle/123456789/28/singlesounds_stereo.zip?sequence=7&isAllowed=y' 
    dest_dir='data2013'     
    downloaddata(datbase_link, dest_dir)
        
        

