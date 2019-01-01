import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D,Input,Dropout
import time
from tensorflow.keras.callbacks import TensorBoard
##import matplotlib.pyplot as plt
import numpy as np
import os,random
import cv2

dir_name = "C:\Info\Transfer Learning\data"
Categories = ['horses','Humans']

training_data = []

def create_training_data():
    for category in Categories:
        path = os.path.join(dir_name,category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_size = 100
                new_array  = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
    random.shuffle(training_data)
    
    X,y = [],[]
    for features,labels in training_data:
        X.append(features)
        y.append(labels)
        
    X = np.array(X).reshape(-1,img_size,img_size,3) 
    return X,y


name = "Cats vs Dogs-cnn-2x64-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir=f'logs/{name}')
#
#
X,y = create_training_data()
#print(X.shape)
X = X/255.0



#model = Sequential()
#model.add(Conv2D(20,(3,3),input_shape=X.shape[1:],name='first_conv_layer'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#
#model.add(Conv2D(30,(3,3),name='second_conv_layer'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#
#model.add(Flatten())
##model.add(Dense(64))
##model.add(Activation('relu'))
#
#model.add(Dense(1,name='dense_layer'))
#model.add(Activation('softmax'))
#
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#
#model.fit(X,y,batch_size=16,epochs=20,validation_split=0.1,callbacks=[tensorboard])
#model.save('Cats_v_Dogs.model')

#def predict_img(filepath):
#    img_size = 50
#    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
#    new_array  = cv2.resize(img_array,(img_size,img_size))
#    return new_array.reshape(-1,img_size,img_size,1)
#
prev_model = tf.keras.models.load_model('Cats_v_Dogs.model') 

top_model =Sequential()
top_model.add(Dense(64,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1,activation='softmax'))

custom_model = Model(inputs=prev_model.input,outputs=top_model(prev_model.output))

custom_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

custom_model.fit(X,y,batch_size=16,epochs=20,validation_split=0.1,callbacks=[tensorboard])

#prediction = model.predict([predict_img('dog.jpg')])
#
#print(Categories[int(prediction[0][0])])





