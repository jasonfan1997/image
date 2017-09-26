
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D,Dropout, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
from keras import applications
import keras
import keras_resnet
import keras_resnet.models
import numpy as np
import tensorflow as tf


def to_one_hot(array):
    array=array.astype(np.int)
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]
    
def show_layers(model):
   for i, layer in enumerate(model.layers):
       print(i, layer.name)

def top_3_categorical_accuracy(y_true, y_pred, k=3):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))
    
    

def train_whole_net():
    '''base_model= keras.applications.inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False, pooling='avg')
    top_model = Sequential()
    top_model.add(Dropout(0.3, input_shape=(1536,)))
    #top_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',top_3_categorical_accuracy])
    #model.add(top_model)
    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))'''
    
    img_width, img_height = 224, 224

    top_model_weights_path = 'resnet18_try1.h5'
    train_data_dir = '../data/scene_classification/scene_train_images_20170904'
    validation_data_dir = '../data/scene_classification/scene_validation_images_20170908'
    #train_labels = np.load('training_labels.npy')
    #validation_labels = np.load('validation_labels.npy')
    nb_train_samples = 53879
    nb_validation_samples = 7120
    epochs = 50
    batch_size = 100
    
    
    shape, classes = (img_height, img_width, 3), 80
    
    x = keras.layers.Input(shape)

    model = keras_resnet.models.ResNet18(x, classes=classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',top_3_categorical_accuracy])
   
    datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    callback=keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
    model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
    keras.models.save_model(model,top_model_weights_path)
#save_features()
train_whole_net()