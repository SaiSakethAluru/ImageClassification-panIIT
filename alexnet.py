from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet_model(img_shape=(224, 224, 3), n_classes=6, l2_reg=0.,
        weights=None):

        # Initialize model
        alexnet = Sequential()

        # Layer 1
        alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
                padding='same', kernel_regularizer=l2(l2_reg)))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        alexnet.add(Conv2D(256, (5, 5), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))
        # Layer 4
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))

        # Layer 5
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same'))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        alexnet.add(Flatten())
        alexnet.add(Dense(3072))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))
        # Layer 7
        alexnet.add(Dense(4096))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 8
        alexnet.add(Dense(n_classes))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('softmax'))

        if weights is not None:
                alexnet.load_weights(weights)

        return alexnet

model=alexnet_model()
model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])


batch_size=8

train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './traindata/train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './traindata/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=4000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=400 // batch_size)
