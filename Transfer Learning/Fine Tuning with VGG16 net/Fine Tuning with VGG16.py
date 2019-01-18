from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,InputLayer
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2,os,random,pickle
import numpy as np

#train_path = 'PetImages/train'
#test_path = 'PetImages/test'
#valid_path = 'PetImages/valid'

#train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(100,100),classes=['dog','cat'],batch_size=20)
#test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(100,100),classes=['dog','cat'],batch_size=4)
#valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(100,100),classes=['dog','cat'],batch_size=10)

training_data = []
Categories = ['Dog','Cat']
dir_name = os.path.dirname(__file__)


def create_training_data():
    for category in Categories:
        path = os.path.join(dir_name,'PetImages',category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_size = 128
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
#   If you need pickling
    
#    pickle_out = open('X.pickle','wb')
#    pickle.dump(X,pickle_out)
#    pickle_out.close()
#    
#    pickle_out = open('y.pickle','wb')
#    pickle.dump(y,pickle_out)
#    pickle_out.close()   

X,y=create_training_data()

#pickle_in_X = open('X.pickle','rb')
#X = pickle.load(pickle_in_X)
#
#pickle_in_y = open('y.pickle','rb')
#y = pickle.load(pickle_in_y)

vgg_model = vgg16.VGG16(weights='imagenet',input_shape=(128,128,3))

#vgg_model.summary()
model =Sequential()
model.add(InputLayer(input_shape=(128,128,3)))
for layer in vgg_model.layers[:-1]:
    model.add(layer)
 
for layer in model.layers:
    layer.trainable =False


model.add(Dense(1,activation='softmax'))
#model.summary()
opt = Adam(lr=1e-4)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#model.fit_generator(train_batches,steps_per_epoch=6,validation_data=valid_batches,validation_steps=6,epochs=10)
model.fit(X,y,batch_size=32,epochs=5,validation_split=0.1)
#model.save('VGG16_TL.model')
#model.predict_generator(test_batches,steps=4)
