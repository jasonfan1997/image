#
# Modified from Keras doc: Fine-tune InceptionV3 on a new set of classes
#Keras blog
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np

img_width, img_height = 299, 299

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '../data/scene_classification/scene_train_images_20170904'
validation_data_dir = '../data/scene_classification/scene_validation_images_20170908'
train_labels = np.load('training_labels.npy')
nb_train_samples = len(train_labels)
nb_validation_samples = 800
epochs = 50
batch_size = 20



# create the base pre-trained model
#base_model = InceptionResNetV2(weights='imagenet', include_top=False)
#keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

datagen = ImageDataGenerator(rescale=1. / 255)

generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True)

input_tensor = Input(shape=(img_width,img_height,3))
# build the VGG16 network
base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)
# add a global spatial average pooling layer

#x = Flatten(input_shape=base_model.output_shape[1:])(base_model.output)
x = Dropout(0.2)(base_model.output)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(80, activation='softmax')(x)
#top_model.load_weights('bootlneck_fc_model.h5')


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
model.summary()

# compile the model (should be done *after* setting layers to non-trainable)
#default: rmsprop
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0005), loss='sparse_categorical_crossentropy')


model.fit_generator(generator, 20)

# train the model on the new data for a few epochs
#model.fit_generator(generator, 1 + nb_train_samples // batch_size)


'''
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
'''