from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D,Dropout, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
from keras import applications
import keras
import numpy as np
import tensorflow as tf


<<<<<<< HEAD
top_model_weights_path = 'top_model_res50_365_one.h5'
train_labels = np.load('training_labels.npy')
validation_labels = np.load('validation_labels.npy')
nb_train_samples = len(train_labels)
nb_validation_samples = 7120
epochs = 10
batch_size = 20
=======

#arch = 'resnet'
arch = 'desnet'



top_model_weights_path = 'top_model_'+arch+'_365_test03.h5'
train_data = np.load(arch+'.npy')
#validation_labels = np.load('validation_labels.npy')
nb_train_samples = len(train_data[:,0])
nb_validation_samples = 7120
epochs = 10
batch_size = 200



>>>>>>> ea8dc5e509c49b1d353d56bf026adfc8fb7d1a2c

def to_one_hot(array):
    array=array.astype(np.int)
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]

def top_3_categorical_accuracy(y_true, y_pred, k=3):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))
def train_top_model():
    np.random.seed(1)
<<<<<<< HEAD
    global train_labels
    global validation_labels
    train_data = np.load('resnet.npy')
    #train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    #train_labels have been defined  globally
    
    validation_data = np.load('resnetv.npy')
=======

    global train_data
    #global validation_labels
    #train_data = np.load(arch+'.npy')
    #train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    #train_labels have been defined  globally
    
    validation_data = np.load(arch+'v.npy')



>>>>>>> ea8dc5e509c49b1d353d56bf026adfc8fb7d1a2c
    #validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
<<<<<<< HEAD
    model.add(Dropout(0.3, input_shape=(2048,)))
    model.add(Dense(1000, activation='relu'))
    '''
    model.add(Dropout(0.3))
    model.add(Dense(80, activation='softmax'))
    '''
=======

    model.add(Dropout(0.3, input_shape=(validation_data.shape[1]-1,)))
    #model.add(Dense(1000, activation='relu'))
    
    #model.add(Dropout(0.2))
    model.add(Dense(80, activation='softmax'))
    


>>>>>>> ea8dc5e509c49b1d353d56bf026adfc8fb7d1a2c
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',top_3_categorical_accuracy])
    #keras.optimizers.RMSprop(lr=0.0005)
    np.random.shuffle(train_data)
    np.random.shuffle(validation_data)
    train_labels=train_data[:,0]
    train_data=train_data[:,1:]
    
    validation_labels=validation_data[:,0]
    validation_data=validation_data[:,1:]  
    #validation_labels=K.one_hot(validation_labels,80)
    validation_labels=to_one_hot(validation_labels.flatten())
    train_labels=to_one_hot(train_labels.flatten())
    #model.load_weights(top_model_weights_path)
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


#save_features()

train_top_model()
