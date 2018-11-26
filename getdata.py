import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

def data_input():
    """the X_train.shape is  (7188, 28)
       the X_test.shape is  (3541, 28)
       the Y_train.shape is  (7188, 2)
       the Y_test.shape is  (3541, 2)
       Returns:X_train,X_test,Y_train,Y_test"""
    data=pd.read_csv("Sentiment.csv")
    data=data[["text","sentiment"]]
    data=data[data.sentiment!="Neutral"]
    data["text"]=data["text"].apply(lambda x:x.lower())#将text的字母全部转化为小写
    data["text"]=data["text"].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    #print(data[ data['sentiment'] == 'Positive'].size)
    #print(data[ data['sentiment'] == 'Negative'].size)
    for idx,row in data.iterrows():#去除数据text中的"rt"字符
        row[0] = row[0].replace('rt',' ')
    max_features=2000
    tokenizer=Tokenizer(num_words=max_features,split=" ")
    tokenizer.fit_on_texts(data["text"].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    #print(X[0])
    X = pad_sequences(X)
    #print(X[0])
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
    #print("the X_train.shape is ",X_train.shape)
    #print("the X_test.shape is ",X_test.shape)
    #print("the Y_train.shape is ",Y_train.shape)
    #print("the Y_test.shape is ",Y_test.shape)
    return X_train,X_test,Y_train,Y_test
    