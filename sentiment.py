from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten,Conv1D,MaxPooling1D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from keras.layers import GlobalMaxPooling1D,Embedding,SpatialDropout1D

from Tkinter import *
import Tkinter as tk
from ttk import *
root = Tk()
#root1=Tk()
max_words = 5000 # We will only consider the 10K most used words in this dataset

def sentimet():

    try:

        model=load_model('/home/harshul/Desktop/senti.h5')
        print('Model loaded')
        y_text=e.get()

        print(y_text)
        df = pd.read_csv("/home/harshul/Desktop/dataset.csv",encoding='latin-1')


        y_text=e.get()

        tokenizer = Tokenizer(num_words=max_words) # Setup
        #tokenizer.fit_on_texts(x_train)
        #tokenizer.fit_on_texts([y_text]) # Generate tokens by counting frequency

        seq = tokenizer.texts_to_sequences([y_text])


        seq = pad_sequences(seq, maxlen=200)
        prediction = model.predict(seq)
        label=tk.Label(root,text='positivity:'+str(prediction))
        label.pack()
        label.config(justify=CENTER,font=("arial",20,"bold"),padx=10,pady=10,background="yellow",foreground="blue")

        print('positivity:',prediction)

    except:
        df = pd.read_csv("/home/harshul/Desktop/dataset.csv",encoding='latin-1')

        x_train=df.iloc[:,0].values
        y_train=df.iloc[:,1].values

        #from nltk.corpus import stopwords
        for i in range (len(x_train)):
            x_train[i]=re.sub(r'\W',' ',x_train[i])
            x_train[i]=re.sub(r'\s+',' ',x_train[i])

        y_text=e.get()

        tokenizer = Tokenizer() # Setup
        tokenizer.fit_on_texts(x_train +[y_text]) # Generate tokens by counting frequency
        sequences = tokenizer.texts_to_sequences(x_train) # Turn text into sequence of numbers
        word_index = tokenizer.word_index
        print(len(word_index))

        x_train=pad_sequences(sequences,maxlen=200,padding='post',truncating='post',value=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

        model=Sequential()
        model.add(Embedding(len(word_index)+1,128,input_length=200))
        model.add(SpatialDropout1D(0.2))

        model.add(Conv1D(64,3,activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.3))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=3,batch_size=256,validation_data=(x_val, y_val),validation_split=0.2)
        model.summary()
        model.save('/home/harshul/Desktop/senti.h5')

        seq = tokenizer.texts_to_sequences([y_text])
        seq = pad_sequences(seq, maxlen=200)
        prediction = model.predict(seq)
        label=tk.Label(root,text='positivity:'+str(prediction))
        label.pack()
        label.config(justify=CENTER,font=("arial",20,"bold"),padx=10,pady=10,background="yellow",foreground="blue")

        print('positivity:',prediction*100.0)


panedwindow = tk.PanedWindow(root, orient = VERTICAL)
panedwindow.pack(fill = BOTH, expand = False)

frame1 = tk.Frame(panedwindow, width = 100, height = 500, relief = SUNKEN)
panedwindow.add(frame1)

label=tk.Label(frame1,text="Write text for sentiment analysis:")
label.pack()
label.config(justify=CENTER,font=("arial",20,"bold"),padx=10,pady=10,background="yellow",foreground="blue")

e=tk.Entry(root,width=50)
e.pack()
e.config(justify=CENTER,font=("arial",20,"bold"))

button=tk.Button(root,text="click")
button.pack()
button.config(justify=CENTER ,font=("arial",20), width=20,command=sentimet)





root.mainloop()
