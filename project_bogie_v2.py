##################################################################################
# Author:   Adam Tipton
# Title:    Project Bogie
# Version:  2
#
# Company:  Brigham Young University - Idaho
# Course:   CSE 499 Senior Project
# Semester: Fall 2020
# 
# Description:
#   Project Bogie is a Convolutional Neural Network written in the Python language. 
#   This CNN will train a model from a dataset the ability to identify military 
#   aircraft/jets. 
#   
#   Once trained, the program will create save the model for use in an application
#   that allows a user to input an image to test if it contains a military aircraft.
#
#   This version will use a 3 block CNN that draws from the VGG16 model. This gives
#   our model a headstart in training as many of the weights are preset, giving 
#   us the advantage of a shortened training time. 
#
#   Project Bogie uses TensorFlow as a backend and Keras as a driving force for training.
#
# Sources:
#   Much inspiration and reusable code base is taken from the following tutorial -
#   URL: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
###################################################################################   
###################################################################################

# System
import sys

# This important import gives us a headstart on training our model.
from keras.applications.vgg16 import VGG16

# Other important keras imports
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# This import is for playing a sound
import winsound

# This allows us to easily adjust epochs
epoch_rate = 50

# The learning rate is an important factor in training neural networks
learning_rate = 0.0055
# Import for pyplot
from matplotlib import pyplot


# The momentum variable can be adjusted to tweak learning
momentum_rate = 0.9

# The batch variable allows us to adjust batch sizes
batch = 80

# The location of our dataset for training
dataSet_dir = 'C:/Users/Adam/source/repos/military_aircraft/aircraft_folders_unsorted'

# The location of an alert sound we'd like to play once it finishes
sound_url = 'C:/Users/Adam/Music/Sounds/Martian_Alert'


##########################################################
################         FUNCTIONS        ################
##########################################################

# This function will create our model. Here we call VGG16. 
def create_model():
    # define a model variable and load the VGG16 information inside it.
    # it defines the expected shape of our files.
    model = VGG16(include_top=False, input_shape=(244, 244, 3))

    # Loop through and set the trainable flags to false for now.
    # We don't need to train them as VGG16 comes pretrained.
    for pre_existing_layer in model.layers:
        pre_existing_layer.trainable = False

    # Here we add new classifier layers to the model, see how they stack?
    # We'll use relu in class1 Dense(), and sigmoid in the output Dense()
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    # New model definition created
    # The inputs are taken from the VGG16 model, the outputs are define above.
    model = Model(inputs=model.inputs, outputs=output)

    # Compile and then return the model
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
    #opt = SGD(lr=learning_rate, momentum=momentum_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Lets plot diagnostic data to see what is going on
def diagnostics(history):
    # plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='red', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='red', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# This function will test and evaluate the model
# It takes a directory string for the sound we'd like to play
def train_and_evaluate_model(sound_url):
    # create the model
    model = create_model()

    # Here we create the data generator
    data_gen = ImageDataGenerator(featurewise_center=True, width_shift_range=[-50, 50], height_shift_range=0.25, 
                                  horizontal_flip=True, vertical_flip=True, rotation_range=15, brightness_range=[0.8, 1.2], zoom_range=[0.2, 1.2])

    # Here we specify the image mean values for centering purposes
    data_gen.mean = [123.68, 116.779, 103.939]

    # Create the iterator for file handling
    training_iterator = data_gen.flow_from_directory(dataSet_dir,
        class_mode='binary', batch_size=batch, target_size=(244, 244))

    #callback
    filepath = "C:/Users/Adam/source/repos/project_bogie_v2/weights.best-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    # Fit the model
    history = model.fit_generator(training_iterator, steps_per_epoch=len(training_iterator), 
        validation_data = training_iterator, epochs=epoch_rate, callbacks=callbacks_list, verbose=0)

    # Here we evalute the model
    _, acc = model.evaluate_generator(training_iterator, steps=len(training_iterator), verbose=0)
    print('Accuracy:')
    print('>%.3f' % (acc * 100.0))
    print(learning_rate)
    print('\n')
    

    # This will produce the learning curve graphs
    print('Summarizing daignostic curves...\n')
    diagnostics(history)
    
    # Now we save the model
    print("Saving Model...")
    model.save('project_bogie_v2_model.h5')

    # Play an alert when the model is done training. Replace the file with your own.
    winsound.PlaySound(sound_url, winsound.SND_FILENAME)
    print("Model now complete...")

# This is the entry point to our program. From here, all of the training begins.
train_and_evaluate_model(sound_url)