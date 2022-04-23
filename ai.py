# import the necessary packages
import time
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def predict(image):
    model = load_model('model.h5', custom_objects = {'top_2_accuracy' : top_2_accuracy, 'top_3_accuracy' : top_3_accuracy})

    image = cv2.resize(image,(224,224))
    image = np.reshape(image,[1,224,224,3])
    image = image/255.0
    
    result = model.predict(image)
    index = np.argmax(result)
    label = ""
    confidence = str(int(result[0][index] * 100)) + "%"
    
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    label = emotions[index]
    return (label, confidence)