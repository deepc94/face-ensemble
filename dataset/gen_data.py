from __future__ import print_function
import os

 
# Set the directory you want to start from
rootDir = './lfw2'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % os.path.basename(dirName))
    for fname in fileList:
        print('\t%s' % fname)