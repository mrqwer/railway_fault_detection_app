import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_model():
    
    model = tf.keras.models.load_model("model/railway.h5")

    return model

def preprocess_image(file_path: str) -> np.ndarray:
    
    img = cv2.imread(file_path)
    plt.imshow(img)
    img = cv2.resize(img,(300,300))
    img = np.reshape(img,[1,300,300,3])

    return img




