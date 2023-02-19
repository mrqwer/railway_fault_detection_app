import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_model():
    model  = tf.keras.models.load_model("model/model_resnet50.h5")
    return model

def preprocess_image(file_path: str) -> np.ndarray:
    
    img = cv2.imread(file_path)
    plt.imshow(img)
    img = cv2.resize(img,(448,448))
    img = np.reshape(img,[1,448,448,3])

    return img


