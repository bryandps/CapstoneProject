import flask
from flask import Flask, render_template, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity

import tensorflow as tf
import tensorflow_hub as hub
import keras

import requests
from requests import get
import urllib.request as req
import os
import PIL
from PIL import Image

from resizeimage import resizeimage

pd.options.display.max_colwidth = 10000

app = flask.Flask(__name__, template_folder='templates', static_url_path='/static/')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


df = pd.read_csv('dogs.csv',index_col=0)
dogs_df = df[['adaptability', 
            'all_around_friendliness', 
            'health_and_grooming_needs', 
            'trainability', 
            'physical_needs',
            'adapts_well_to_apartment_living', 
            'good_for_novice_owners', 
            'sensitivity_level', 
            'tolerates_being_alone', 
            'tolerates_cold_weather', 
            'tolerates_hot_weather', 
            'affectionate_with_family', 
            'kid_friendly_dogs', 
            'dog_friendly', 
            'friendly_towards_strangers', 
            'amount_of_shedding', 
            'drooling_potential', 
            'easy_to_groom', 
            'general_health', 
            'potential_for_weight_gain', 
            'size', 
            'easy_to_train', 
            'intelligence', 
            'potential_for_mouthiness', 
            'prey_drive', 
            'tendency_to_bark_or_howl', 
            'wanderlust_potential',
            'energy_level',
            'intensity',
            'exercise_needs',
            'potential_for_playfulness']]

def overall_recommender(breed,dist='cosine'):
    '''
    Input: Name of breed (string)
    Output: 5 Breeds with most similar temperaments according to dogtime.com ratings
    '''
    y = dogs_df.loc[[breed],:]
    euc_dists = euclidean_distances(dogs_df.values,y.values)
    euc_ind = np.argsort(euc_dists.flatten())
    cos_dists = cosine_similarity(dogs_df.values,y.values)
    cos_ind = np.argsort(cos_dists.flatten())
    if dist == 'euclidean':
        return [dogs_df.iloc[ind,:].name for ind in euc_ind][1:6]
    elif dist == 'cosine':
        return [dogs_df.iloc[ind,:].name for ind in cos_ind][-1:-6:-1]

def predictions_recommender(breed,photo_list,dist='cosine'):
    '''
    Input: Name of breed (string), List of dogs you're considering (list)
    Output: Ordered list starting from most similar to least
    '''
    y = dogs_df.loc[[breed],:]
    X = dogs_df.loc[photo_list,:]
    euc_dists = euclidean_distances(X.values,y.values)
    euc_ind = np.argsort(euc_dists.flatten())
    cos_dists = cosine_similarity(X.values,y.values)
    cos_ind = np.argsort(cos_dists.flatten())
    if dist == 'euclidean':
        return [X.iloc[ind,:].name for ind in euc_ind]
    elif dist == 'cosine':
        return [X.iloc[ind,:].name for ind in cos_ind][::-1]

def compatibility_score(profile, breed, dist='cosine'):
    '''
    Input: Profile created from radio inputs (np array)
    Output: 5 Breeds with most similar temperaments according to dogtime.com ratings
    '''
    y = profile
    euc_dists = euclidean_distances(dogs_df.loc[(dogs_df.index == breed)].values,y)
    euc_ind = np.argsort(euc_dists.flatten())
    cos_dists = cosine_similarity(dogs_df.loc[(dogs_df.index == breed)].values,y)
    cos_ind = np.argsort(cos_dists.flatten())
    if dist == 'euclidean':
        return euc_dists
    elif dist == 'cosine':
        return cos_dists

def profile_recommender(profile, breed, dist='cosine'):
    '''
    Input: Profile created from radio inputs (np array)
    Output: 5 Breeds with most similar temperaments according to dogtime.com ratings
    '''
    score = compatibility_score(profile, breed)
    if score >= 0.8:
        return True
    else:
        
        y = profile
        euc_dists = euclidean_distances(dogs_df.values,y)
        euc_ind = np.argsort(euc_dists.flatten())
        cos_dists = cosine_similarity(dogs_df.values,y)
        cos_ind = np.argsort(cos_dists.flatten())
        if dist == 'euclidean':
            return [dogs_df.iloc[ind,:].name for ind in euc_ind][1:6]
        elif dist == 'cosine':
            return [dogs_df.iloc[ind,:].name for ind in cos_ind][-1:-6:-1]

def load_model(model_path):
    '''
    Loads a saved model from a specified path.
    
    Parameter:
    - model path = location of the saved .hdf5 model
    
    Returns:
    - Loaded model
    
    '''
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
    return model

### This code block converts all images into tensors and in the desired dimension

### Standard size for image classification is 224
IMG_SIZE = 224

def process_image(image_path):
    '''
    This function takes in an image then resizes and converts it into
    Tensors and desired dimensions.
    
    Parameters:
    - image_path = image
    
    Output:
    - Resized image in tensor format
    
    '''
    ### Read in image file
    image = tf.io.read_file(image_path)
    
    ### Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)

    ### Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
  
    ### Resize the image to desired size (224, 244)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image

def get_image_label(image_path, label):
    '''
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label)
  
    Parameters:
    - image_path = where the image is located
    - label = corresponding image class
  
    Returns:
    - images and labels resized and converted into tensors
    '''
    image = process_image(image_path)
    return image, label

### Define the batch size, 32 is a good default
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    '''
    Creates batches of data out of image (features) and label (target) pairs.
    Shuffles the data if it's training data.
  
    Parameters:
    - x = features/images
    - y = target
    - batch_size = size of batch (32 is standard)
    - valid_data = whether for validation

    '''
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch
  
    # If the data if a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        # If the data is a training dataset, we shuffle it
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    
        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(x))

        # Create (image, label) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)

        # Turn the data into batches
        data_batch = data.batch(BATCH_SIZE)
    return data_batch

def get_pred_label(prediction_probabilities):
    '''
    Turns an array of prediction probabilities into a label.
    
    Parameters:
    - prediction probabilities
    
    Returns:
    - actual breed predicted
    '''  

    return unique_breeds[np.argmax(prediction_probabilities)]

def inverse(num):
    if num == 5:
        return 1
    elif num == 4:
        return 2
    elif num == 3:
        return 3
    elif num == 2:
        return 4
    else:
        return 5
### Get a predicted label based on an array of prediction probabilities
# pred_label = get_pred_label(predictions[0])
# pred_label
unique_breeds = pd.read_csv('unique_breeds.csv', index_col = 0)
unique_breeds = unique_breeds['0']

columns = list(dogs_df.columns)



# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        dog_url = flask.request.form['dog_image']

        profile = dogs_df.describe().T['mean'].values

        profile[np.array([4, 27, 28, 29, 30 ])] = flask.request.form['exercise_needs']
        profile[19] = inverse(flask.request.form['exercise_needs'])
        profile[np.array([0, 5, 6, 20])] = flask.request.form['apartment_ready']
        profile[11] = flask.request.form['affection']
        profile[np.array([2, 15, 16, 17])] = flask.request.form['fur_drool']
        profile[np.array([3, 21])] = flask.request.form['trainability']
        profile[np.array([23, 25])] = inverse(flask.request.form['trainability'])
        profile[np.array([13, 14])] = flask.request.form['friendliness']
        profile[7] = inverse(flask.request.form['friendliness'])
        profile[np.array([1, 12])] = flask.request.form['kids']
        profile[22] = flask.request.form['intelligence']
        profile[8] = flask.request.form['tolerates_alone']
        profile[np.array([24, 26])] = inverse(flask.request.form['tolerates_alone'])
        profile[9] = flask.request.form['climate']
        profile[10] = inverse(flask.request.form['climate'])
        profile[18] = flask.request.form['finance']
        profile = profile.reshape(1,-1)

        req.urlretrieve(dog_url, "predict/" + 'predict.jpg')
        req.urlretrieve(dog_url, "static/" + 'predict.jpg')


        img = Image.open('static/predict.jpg')
        new_img = img.resize((50,50))
        new_img.save("predict.jpg", "JPEG", optimize=True) 
        
        os.system('find . -name ".DS_Store" -delete')
        
        # Load our model trained on 9500+ images
        model_mobilenet = load_model('models/weights.best.mobilenet.hdf5')

        ### Get custom image filepaths      
        custom_path = "predict/"
        custom_image_paths = [custom_path + fname for fname in os.listdir(custom_path)]
        #custom_image_paths = custom_path

        custom_data = create_data_batches(custom_image_paths, test_data=True)
        custom_preds = model_mobilenet.predict(custom_data)

        breed = get_pred_label(custom_preds[0])

        verdict = profile_recommender(profile, breed)

        if verdict == True:
            df = pd.read_csv('dogs.csv',index_col=0)
            df = df.loc[(df.index == breed)]
            highlights = df['highlights'].to_string()
            highlights = highlights.split(" ")[2:]
            highlights = ' '.join(highlights)
            personality = df['personality'].to_string()
            personality = personality.split(" ")[2:]
            personality = ' '.join(personality)
            size = df['size_description'].to_string()
            size = size.split(" ")[2:]
            size = ' '.join(size)

            adoption_df = pd.read_csv('adoption_dogs.csv', index_col = 0)
            #adf = adoption_df.loc[(adoption_df['breed'].str.contains(breed.title()))]
            adf = adoption_df.loc[(adoption_df['breed'] == breed)]
            name1 = adf.iloc[0]['name']
            sex1 = adf.iloc[0]['sex']
            age1 = adf.iloc[0]['age']
            breed1 = adf.iloc[0]['breed']
            link1 = adf.iloc[0]['link']
            name2 = adf.iloc[1]['name']
            sex2 = adf.iloc[1]['sex']
            age2 = adf.iloc[1]['age']
            breed2 = adf.iloc[1]['breed']
            link2 = adf.iloc[1]['link']
            name3 = adf.iloc[2]['name']
            sex3 = adf.iloc[2]['sex']
            age3 = adf.iloc[2]['age']
            breed3 = adf.iloc[2]['breed']
            link3 = adf.iloc[2]['link']
            name4 = adf.iloc[3]['name']
            sex4 = adf.iloc[3]['sex']
            age4 = adf.iloc[3]['age']
            breed4 = adf.iloc[3]['breed']
            link4 = adf.iloc[3]['link']
            name5 = adf.iloc[4]['name']
            sex5 = adf.iloc[4]['sex']
            age5 = adf.iloc[4]['age']
            breed5 = adf.iloc[4]['breed']
            link5 = adf.iloc[4]['link']

            return flask.render_template('positive.html',breed=breed, dog_url=dog_url, highlights=highlights, personality=personality, size=size,
                name1 = name1,
                sex1 = sex1,
                age1 = age1,
                breed1 = breed1,
                link1 = link1,
                name2 = name2,
                sex2 = sex2,
                age2 = age2,
                breed2 = breed2,
                link2 = link2,
                name3 = name3,
                sex3 = sex3,
                age3 = age3,
                breed3 =breed3,
                link3 = link3,
                name4 = name4,
                sex4 = sex4,
                age4 = age4,
                breed4 = breed4,
                link4 = link4,
                name5 = name5,
                sex5 = sex5,
                age5 = age5,
                breed5 = breed5,
                link5 = link5
            )
        
        else:
            breed1 = verdict[0]
            breed2 = verdict[1]
            breed3 = verdict[2]
            breed4 = verdict[3]
            breed5 = verdict[4]
            return flask.render_template('negative.html', breed=breed, dog_url=dog_url, breed1=breed1, breed2=breed2, breed3=breed3, breed4=breed4, breed5=breed5)
           
if __name__ == '__main__':
    app.run(debug=True)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response