#!/usr/bin/env python3
import os
import pandas as pd


def getFiles(dataDir):
    files = os.listdir(dataDir)
    filesList = []
    for fileNm in files:
        if fileNm.split('.')[-1] == 'xls':
            print(fileNm)
            filesList.append(fileNm)
    return filesList


dataDir = './'
filesList = getFiles(dataDir)
for filename in filesList:
    print(filename)
    df = pd.read_excel(filename)
    df.to_excel(filename.split('.')[0]+'.xlsx',
                index=False, engine='xlsxwriter')
