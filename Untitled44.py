#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import librosa
import librosa.display


# In[8]:


import pandas as pd


# In[9]:


meta=pd.read_csv('esc50.csv')


# In[10]:


meta.head()


# In[11]:


meta['category'].value_counts()


# In[12]:


import os
import numpy as np


# In[13]:


audio_file='C:\\Users\\aneri\\OneDrive\\Desktop\\ESC-50-master\\audio'


# In[14]:


audio_file


# In[15]:


def feature_extractor(file):
    audio, sample_rate=librosa.load(file_name,res_type='kaiser_fast')
    mfcc_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfcc_scaled=np.mean(mfcc_features.T,axis=0)
    return mfcc_scaled


# In[16]:


from tqdm import tqdm


# In[17]:


extracted_feature=[]
for index_num,row in tqdm(meta.iterrows()):
    file_name=os.path.join(os.path.abspath(audio_file),str(row['filename']))
    final_class=row['category']
    data=feature_extractor(file_name)
    extracted_feature.append([data,final_class])


# In[18]:


df=pd.DataFrame(extracted_feature,columns=['features','class'])


# In[19]:


df


# In[38]:


X=np.array(df['features'].tolist())
X.shape


# In[21]:


y=np.array(df['class'].tolist())
y.shape


# In[22]:


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
y.shape


# In[43]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[44]:


X_train.shape


# In[45]:


X_test.shape


# In[46]:


y_train.shape


# In[47]:


y_test.shape


# In[48]:


import tensorflow as tf


# In[49]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# In[50]:


model=Sequential()
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('softmax'))


# In[51]:


model.summary()


# In[52]:


#model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[53]:


from tensorflow import keras
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[54]:


from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',verbose=1, save_best_only=True)

start = datetime.now()

model.fit(X_train, y_train, batch_size=32, epochs=500, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[57]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

