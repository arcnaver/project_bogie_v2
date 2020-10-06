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

