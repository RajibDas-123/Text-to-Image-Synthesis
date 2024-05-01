from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.stats import entropy
from scipy.linalg import sqrtm


def load_and_preprocess_image(image_path):
    # Load the image file, resizing it to 299x299 pixels
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = preprocess_input(img)  # Preprocess the image using InceptionV3's required preprocessing
    return img

def inception_score(images, model, batch_size=32):
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        preds.append(model.predict(batch))
    preds = np.concatenate(preds, axis=0)
    # Calculate the marginal probability
    marginal_prob = np.mean(preds, axis=0)
    # Calculate the KL divergence and then the score
    scores = entropy(preds, marginal_prob, base=2, axis=1)
    return np.exp(np.mean(scores))

def calculate_fid(model, real_images, generated_images):
    act1 = model.predict(real_images)
    act2 = model.predict(generated_images)
    # Calculate mean and covariance
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    # Calculate the difference and trace for FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid