from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os
import numpy as np
import pandas as pd


dates = pd.date_range('2007-01-02', periods=2514, freq='D')
a = np.zeros((2514,3))
fTable = pd.DataFrame(a)
fTable.columns = [['Dates','ArticleCount','ApplCount']]
fTable.index = dates

# print(fTable)
for i in range(2007,2014):
    datapathRoot = 'OriData/'+str(i)+'/'
    os.mkdir('PostData/'+str(i))
    postDataPathRoot = 'PostData/'+str(i)+'/'
    datapathSuffix = os.listdir(datapathRoot)
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    for dateSpecific in datapathSuffix:
        files = os.listdir(datapathRoot+dateSpecific)
        fTable['Dates'][dateSpecific] = dateSpecific
        fTable['ArticleCount'][dateSpecific] = len(files)
        n = 0
        os.mkdir(postDataPathRoot + dateSpecific)
        
        for file in files:
            fileName = file.split('-')

            if 'apple' in fileName:
                f = open(datapathRoot+dateSpecific+'/'+file,'r',encoding='windows-1252').read()
                word_tokens = tokenizer.tokenize(f)
                wordProcessed = [w.lower() for w in word_tokens if w not in stop_words]
                text = ' '.join(wordProcessed)
                newFile = open(postDataPathRoot+dateSpecific+'/'+file,'w',encoding='windows-1252').write(text)
                n = n+1
        fTable['ApplCount'][dateSpecific] = n

fTable.to_csv('fTable.csv')



