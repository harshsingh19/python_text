# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:34:41 2018

@author: harsh
"""
print("Importing File...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re, io, sys, time, collections
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print("Importing Completed...")
#getting dataset
print("Getting Dataset...")
dataset = pd.read_csv('C:\\train.tsv',encoding='latin-1', delimiter = '\t', quoting = 2)
print("Sucessfully Completed geting file...")
data   = []
train_labels = []
test_data    = []
test_labels  = []
train_dataset= []
test_dataset = []
print("Getting StopWords...")
cachedStopWords = stopwords.words("english") 
stemmer         = SnowballStemmer("english", ignore_stopwords = True)
print("Getting StopWords Completed")
#cleaning the texts for train dataset
a=len(dataset.index) #length of dataset calculating
print ("Cleaning of the data")
print ("Loading...")
for i in range(0, a):
    review = dataset['SentimentText'][i]
    review = review.lower()
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',review ) ##URLS
    review = re.sub('<.*?>',' ', review) ##html tags"""
    review = re.sub(r'[\w\.-]+@[\w\.-]+',r' ', review)##email addresses
    review = re.sub('@[^\s]+',' ', review) ##Authors
    review = re.sub(r'#([^\s]+)', r'\1',review) ##hash tags
    review = re.sub('\W+',' ',review)  ##special char
    review = re.sub(r'^[^a-zA-Z]',r' ',review)##non words
    review = re.sub(r"\s+", " " ,review,flags=re.UNICODE)## remove multiply space and unicode
    review = re.sub(r'\b(\w+)( \1\b)+', r'\1',review ) ##repetitive words
    review = ' '.join([word for word in review.split() if word not in cachedStopWords]) ##stopwords
    review = " ".join([stemmer.stem(j) for j in review.split()]) #removing unessary words from the list and steming it steming means if the word in list is loved it will convert the word to love as a reaction joining the list to make a sentence again
    review = re.sub('[^a-zA-Z]', ' ',review)
    data.append(review) #writing the values in corpus
print("Cleaning Completed...")
labels = dataset['Sentiment'].values
print("Spliting data into Test and Train dataset \n Loading...")
train_dataset, test_dataset ,train_labels,test_labels= train_test_split(data,labels,test_size=0.2,random_state = 0)
print ("Completed Splited Data into train and test dataset")
#cleaning the texts for test dataset

"""b=len(test_dataset.index) #length of dataset calculating
for i in range(0, b):
    review = ' '.join([word for word in test_dataset['SentimentText'][i].split() if word not in cachedStopWords]) ##stopwords
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',review ) ##URLS
    review = re.sub('<.*?>',' ',review) ##html tags
    review = re.sub(r'[\w\.-]+@[\w\.-]+',r' ',review)##email addresses
    review = re.sub('@[^\s]+','CLEAN_AUTHOR',review) ##Authors
    review = re.sub(r'#([^\s]+)', r'\1',review) ##hash tags
    review = re.sub('\W+',' ',review)  ##special char
    review = re.sub(r'^[^a-zA-Z]',r' ',review)##non words
    review = re.sub(r"\s+", " " ,review,flags=re.UNICODE)## remove multiply space and unicode
    review = re.sub(r'\b(\w+)( \1\b)+', r'\1',review ) ##repetitive words
    review = " ".join([stemmer.stem(j) for j in review.split()]) #removing unessary words from the list and steming it steming means if the word in list is loved it will convert the word to love as a reaction joining the list to make a sentence again
    test_data.append(review)
"""
# Creating the Bag of Words model for train dataset
# Vectorization for text in features
print("Creating text to Vector form ")
print("Loading...")
t0                = time.time()
bigram_vectorizer = CountVectorizer(ngram_range=(1,4),token_pattern=r'\b\w+\b', min_df=75)#making object of class of CountVectorizer. here max feature is used to use max no vector coloums 
analyze           = bigram_vectorizer.build_analyzer()
train_vectors     = bigram_vectorizer.fit_transform(train_dataset).toarray()#fit as array in the matrix
test_vectors      = bigram_vectorizer.transform(test_dataset).toarray()#fit as array in the matrix 0,1 matrix
t1                = time.time()
time_feature      = t1-t0
print ("Completed Creating text to Vector form")
print("Time taken to Vectorize Data",time_feature,"Sec")
# Perform classification with RandomForest
print("Start of Algorithum")
classifier_rf = RandomForestClassifier(n_estimators=50)
print("Training Model with Random Forest Algorithum")
print ("Loading...")
t0 = time.time()
classifier_rf.fit(train_vectors, train_labels)
t1 = time.time()
print("Training Model with Random Forest Algorithum Completed...")
# Predicting the Test set results
print("Predicting values with Random Forest Algorithum")
print ("Loading...")
prediction_rf = classifier_rf.predict(test_vectors)
t2 = time.time()
print ("Completed...")
time_rf_train = t1-t0
print("Time taken to train data",time_rf_train,"Sec")
time_rf_predict = t2-t1
print("Time taken to predict data",time_rf_predict,"Sec")
# Predicting the Test set probablity
prediction_probs = classifier_rf.predict_proba(test_vectors) 
cm = confusion_matrix(test_labels, prediction_rf)# Making the Confusion Matrix
x=cm[0][0]+cm[1][1]
print("The total no of correct prediction",x)
y=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
print("The total no of wrong prediction",y)
prob = x/y
print ("The probality of getting correct value",prob)


'''# Perform classification with SVM, kernel=rbf
#(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.0, kernel='rbf', max_iter=-1, probability=False,random_state=None, shrinking=True, tol=0.001, verbose=False)
classifier_rbf = svm.SVC(kernel='rbf',C=1.0,cache_size=500,class_weight=None,coef0=0.0, degree=3, gamma=5.0, max_iter=-1, probability=True,random_state=5, shrinking=True, tol=0.00001, verbose=False)
#classifier_rbf   = svm.SVC(kernel='rbf',C=200.0,Probablity=True)
t0               = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1               = time.time()
prediction_rbf   = classifier_rbf.predict(test_vectors)
t2               = time.time()
time_rbf_train   = t1-t0
time_rbf_predict = t2-t1
prediction_probs = classifier_rbf.predict_proba(test_vectors) 

with open("D:\\result.txt", "w") as outfile:
    out_string=""
    for i in range(len(prediction_rbf)):
        out_string+="\n"
        out_string+=test_data[i]
        out_string+="FIELDSEPARATOR "
        out_string+=prediction_rbf[i]
        out_string+=" FIELDSEPARATOR"
        out_string+=str(prediction_probs[i])
    outfile.write(out_string)

'''
'''
# # Perform classification with GLM 
# classifier_glm = linear_model.LinearRegression()
# t0 = time.time()
# classifier_glm.fit(train_vectors, train_labels)
# t1 = time.time()
# prediction_glm = classifier_rbf.predict(test_vectors)
# t2 = time.time()
# time_glm_train = t1-t0
# time_glm_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC(C=.91, penalty="l1", dual=False)
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels_train)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)

t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

# Perform classification with NN uniform
n_neighbors = 10
classifier_nn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
t0 = time.time()
classifier_nn.fit(train_vectors, train_labels_train)
t1 = time.time()
prediction_nn = classifier_nn.predict(test_vectors)
t2 = time.time()
time_nn_train = t1-t0
time_nn_predict = t2-t1


# Perform classification with NN distance
classifier_nnd = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
t0 = time.time()
classifier_nnd.fit(train_vectors, train_labels_train)
t1 = time.time()
prediction_nnd = classifier_nnd.predict(test_vectors)
t2 = time.time()
time_nnd_train = t1-t0
time_nnd_predict = t2-t1
'''
"""# Predicting the Test set results
#with open("D:\\result.txt", "w") as outfile:                                                                                     
 #   out_string=""                                                                                                                              
for i in range(len(prediction_rf)):                                                                                                       
        #out_string+="\n"                                                                                                                       
        #out_string+=  
    print (test_data[i])                                                                                                       
        #out_string+=  "FIELDSEPARATOR "                                                                                                                        
    print (prediction_rf[i])                                                                                                        
        #out_string+= " FIELDSEPARATOR"                                                                                                                        
    print (str(prediction_probs[i]))                                                                                                   
    #outfile.write(out_string)
    #outfile.close()"""
