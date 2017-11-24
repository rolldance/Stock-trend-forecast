import os
from textblob import TextBlob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime,timedelta
# def sentimentData():

a = np.zeros((2514,3))    #去除最后一行
a.round(4)
df = pd.DataFrame(a)
df.columns=[['Polarity','Subjectivity','WordMentions']]
dates = pd.date_range('2007-01-02', periods=2514, freq='D')
df.index = dates

sub_sentiment=[]
sub_date=[]
sub_subjectivity=[]
sub_wordmentions=[]

newDate=[]
newSentiment=[]
newSubjectivity = []
newWordMentions = []
date1=datetime(2007,1,2)
for i in range(2514):
    newDate.append(float(date1.strftime('%Y%m%d')[2:]))
    # c.append(float(date1.strftime('%Y%m%d')))
    date1 = date1+timedelta(days=1)

print(newDate)
print(len(newDate))
for i in range(2007, 2014):
    datapathRoot = '../financialnewsData/PostData/'+str(i)+'/'
    datapathSuffix = os.listdir(datapathRoot)
    for dateSpecific in datapathSuffix:
        polaritySub = []
        subjectiveSub = []
        frequenciesSub = []
        files = os.listdir(datapathRoot+dateSpecific)
        # print(files)
        for file in files:
           if file != '':
               f = open(datapathRoot+dateSpecific+'/'+file, 'r', encoding='windows-1252').read()
               wiki = TextBlob(f)
               wiki.lower()
               polaritySub.append(wiki.sentiment.polarity)
               subjectiveSub.append(wiki.sentiment.subjectivity)
               frequenciesSub.append(wiki.words.count('apple'))
        if len(polaritySub) != 0:

           sub_sentiment.append(np.array(polaritySub).mean())
           sub_subjectivity.append(np.array(subjectiveSub).mean())
           sub_wordmentions.append(np.array(frequenciesSub).mean())
           sub_date.append(float(dateSpecific[2:]))

 # print(sub_date,sub_sentiment)
f_newSentiment = interp1d(sub_date,sub_sentiment,kind='cubic')
f_newSubjectivity = interp1d(sub_date,sub_subjectivity,kind='cubic')
f_newWordMentions = interp1d(sub_date,sub_wordmentions,kind='cubic')
for i in range(len(newDate)):
    # print(newDate[i])
    newSentiment.append(f_newSentiment(newDate[i]))
    newSubjectivity.append(f_newSubjectivity(newDate[i]))
    newWordMentions.append(f_newWordMentions(newDate[i]))


df['Polarity'] = newSentiment
df['Subjectivity'] = newSubjectivity
df['WordMentions'] = newWordMentions
# print(df)
df['key'] = df.index
df.to_csv('newSenti.csv')







