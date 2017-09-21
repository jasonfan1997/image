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


top_model_weights_path = 'top_model_res50_365.h5'
train_labels = np.load('training_labels.npy')
validation_labels = np.load('validation_labels.npy')
nb_train_samples = len(train_labels)
nb_validation_samples = 7120
epochs = 10
batch_size = 1000

def to_one_hot(array):
    array=array.astype(np.int)
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]


def train_top_model():
    np.random.seed(1)
    global train_labels
    global validation_labels
    train_data = np.load('bottleneck_features_train.npy')
    #train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    #train_labels have been defined  globally
    
    validation_data = np.load('bottleneck_features_validation.npy')
    #validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dropout(0.3, input_shape=(1536,)))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(80, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',top_3_categorical_accuracy])
    #keras.optimizers.RMSprop(lr=0.0005)
    train_labels=train_labels.reshape((len(train_labels),1))
    trainmerge=np.append(train_data,train_labels,axis=1)
    validation_labels=validation_labels.reshape((len(validation_labels),1))
    
    validation_data=validation_data[:-20,:]
    validationmerge=np.append(validation_data,validation_labels,axis=1)
    np.random.shuffle(trainmerge)
    np.random.shuffle(validationmerge)
    train_data=trainmerge[:,:-1]
    train_labels=trainmerge[:,-1]
    validation_data=validationmerge[:,:-1]
    validation_labels=validationmerge[:,-1]  
    #validation_labels=K.one_hot(validation_labels,80)
    validation_labels=to_one_hot(validation_labels.flatten())
    train_labels=to_one_hot(train_labels.flatten())
    model.load_weights(top_model_weights_path)
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


#save_features()

train_top_model()
