# Part 1 - Building the CNN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers

# Initialsing the CNN
classifier = Sequential()

# Step 1 - Convolution Layer 
classifier.add(Conv2D(32, (3,  3), input_shape = (500, 500, 3), activation = 'relu'))

#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding second convolution layer
classifier.add(Conv2D(32, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Adding 3rd Concolution Layer
classifier.add(Conv2D(64, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(2, activation = 'softmax')) #number of classes = 3

#Compiling The CNN
classifier.compile(
              optimizer = optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'Maindataset/MainTrain',
        target_size=(500, 500),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'Maindataset/Maintest',
        target_size=(500, 500),
        batch_size=32,
        class_mode='categorical')
model = classifier.fit_generator(
                        training_set,
						steps_per_epoch = 50, #steps per epoch is no. of samples/batch size
						nb_epoch = 20,
						)

# Saving the model
classifier.save('Trained_model.h5')