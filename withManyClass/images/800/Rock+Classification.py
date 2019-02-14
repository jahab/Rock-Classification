

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os
import cv2
import joblib

# In[13]:


DATADIR="/home/jafar/Desktop/Master/jafar/Sem4/Rock Classification/withManyClass/images/800"
#CATEGORIES=["Aaaa", "interflw","pahoehoe","transi"]
#CATEGORIES=["Aaa_enta","Aaa_lava ball","inter_brown","inter_red","pa_domed vesicle","pa_gas cavity","pa_inflection1","pa_inflection2","pa_joints","pa_pipes","pa_pipes and vesicles","pa_ropes","pa_squeezeup","pa_sq up","pa_toes","pa_vesicle banding","pa_vesicle cyl","tr_ftb 800","tr_rubbly","tr_slabby"]

CATEGORIES=[
'Aaa_enta','Aaa_lava ball',
'inter_brown','inter_red',
'pa__hummocky_inflection1','pa_hummocky_inflection2','pa_hummocky_ropes',
'pa_hummocky_squeezeup','pa_hummocky_toes',
'pa_sheet_domed vesicle','pa_sheet_gas cavity','pa_sheet_joints',
'pa_sheet_pipes and vesicles','pa_sheet_sq up','pa_sheet_vesicle banding','pa_sheet_vesicle cyl',
'tr_ftb 800','tr_rubbly','tr_slabby']

numclass=len(CATEGORIES)
# In[14]:


for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break


# In[15]:


IMG_SIZE=300
new=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new)
plt.show()


# In[16]:


#img_array


# In[17]:


training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            #print(np.shape(img_array) ,class_num)
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])

create_training_data()


# for i in training_data:
#     #print(i)
#     height, width = i.shape[:2]
#     break

# In[18]:


import random
random.shuffle(training_data)
X=[]
y=[]
for features, label in training_data:
    X.append(features)
    y.append(label)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

Y=pd.Series(y)

class_counts=Y.value_counts()
print(class_counts.index)
cats=pd.Series(CATEGORIES).iloc[class_counts.index]

fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

#plt.hist(y,bins=numclass, color='#0504aa',alpha=0.7, rwidth=0.85)

plt.scatter(cats,class_counts,s=100)
for i,j in zip(cats,class_counts):
    ax.annotate(str(j),xy=(i,j))

plt.xticks(rotation=90)
plt.title("Plot to show Class Imbalance | Number of images in every class")
plt.show()

pickle_out=open("X.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

X=np.divide(X,255)

# In[33]:


def rock_classifier():
    model=Sequential()

    model.add(Conv2D(20,(2,2),input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(2,2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(2,2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(2,2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(2,2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(100))

    model.add(Dense(numclass))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    #model.fit(X,y,batch_size=1,epochs=30, validation_split=0.1)
    return model
#rock_classifier()


# In[34]:


#X.shape[0:]


# ### CV accuracy




from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline




kfold = KFold(n_splits=5, shuffle=True)


# In[44]:


estimator = KerasClassifier(build_fn=rock_classifier, epochs=25, batch_size=1,verbose=1)
results=cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

filename='results_800_5.sav'
joblib.dump(results, filename)
