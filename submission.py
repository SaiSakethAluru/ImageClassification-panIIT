from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.regularizers import l2
import numpy as np

# Define CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224,224,3), kernel_regularizer=l2(0)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="tf"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="tf"))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering="tf"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="tf"))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './data/traindata/train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224 x 224
        batch_size=batch_size,
        class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './data/traindata/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

# train model
model.fit_generator(
        train_generator,
        steps_per_epoch=4000 // batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=400 // batch_size)

model.save_weights('ResNet_try.h5')  # save weights after training for future use

# data to predict
predict_gen = predict_datagen.flow_from_directory(
        './testdata',           # test data directory
        target_size = (224,224),            # similar preprocessing
        batch_size = batch_size,
        shuffle=False,
        class_mode = None)             # no labels given

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

probabilities = model.predict_generator(predict_gen)    # make predictions
predicted_classes=probabilities.argmax(axis=-1)+1       # get predicted class from one hot encoded vectors
import csv
classes=[]
files=predict_gen.filenames         # get prediction filenames for image id

for i in range(len(predicted_classes)):
    classes.append((int(files[i].split('/')[1].split('.')[0]),int(predicted_classes[i])))   # append(id,category) pairs
classes.sort(key=lambda x:x[0])     # sort by id

# write to csv file
csv_data = [('id','category')]+classes 

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)