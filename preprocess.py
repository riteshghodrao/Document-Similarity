from stemming.porter2 import stem
import os
import string
import shutil
import re
import scipy
import math
import numpy as np
from gensim import corpora, models, similarities
import collections

path = os.getcwd()+'/data/' #path to the dataset

#path to the output of stemmed data set
if os.path.exists('stemmed_output/'):
    shutil.rmtree(os.getcwd()+'/stemmed_output/')
os.mkdir(os.getcwd()+'/stemmed_output/')


stoplist = [] # contains the list of stopwords
f=open('stopword.txt')
for line in f:
    line = line[:-2] # the line is like "the\r\n"
    stoplist.append(line)
f.close()


def preprocess(line):
    line = line.strip()
    line = line.lower()
    line=re.sub("<.*?>","",line)
    for c in string.punctuation:
        line=line.replace(c,' ')
    line2=''  # contains the stemmed sentence 
    line_list = []
    for word in line.split():
        if word in stoplist:
            continue
        if len(word) < 3:
            continue
        stemmed_word = stem(word)
        line2+=stemmed_word+' '
        line_list.append(stemmed_word)
    return line2


for filename in os.listdir(path):
    file1 = open(path+filename,'r')
    file2 = open(os.getcwd()+'/stemmed_output/'+filename, 'w')
    text1 = file1.readlines()
    for line in text1:
        # preprocessing the text
        line2 = preprocess(line)
        file2.write(line2)
