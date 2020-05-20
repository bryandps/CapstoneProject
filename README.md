# What's that Doggie in the Window?
By: **Bryan Santos**

# Social Case

Have you ever wanted to know the breed of dogs you see in social media or in parks because you found they look suitable for yourself?

This project aims to build an application that lets users upload images of a dog and get its breed. The application will then assess the breed characteristics if it is suitable for the user based on lifestyle. If it is, then the system will redirect the user to a dog of that particular breed that is up for adoption.

The project will utilize multi-class image classification and recommendation systems machine learning models to achieve its goals.

![](figures/pet1.png)

The pet industry is a multi-billion dollar industry even just in the United States alone. The trend of owning pets is on a steady rise. Unfortunately, so do the number of dogs that would be without a permanent home or that would be euthanized. Only 1 in 10 dogs will have permanent home. 

![](figures/pet2.png)

Many people buy dogs because of fad or appearances and abandon them, most likely because they do not realize that dogs of different breeds have unique characteristics and may not necessarily match their lifestyles. There is a need to match individuals with specific breeds and promote adoption of compatible breeds.

# Dataset

## Images

There are a total of 28,666 dog images collected through API calls and manual download. The sources of the images are www.dog.ceo, a site that provides API endpoints which returns images of various dog breeds. In addition, I downloaded the dog images dataset from a Udacity Nanodegree. In total, I have images for 173 different breeds.

## Dog Breed Characteristics

45 different breed characteristics of 359 dog breeds were scraped from www.dogtime.com. 31 out of the 45 characteristics are numeral ratings from one to five stars. This dataset will not be used yet in the MVP version of the project.

## Dogs for Adoption

Details of 60,000+ actual dogs for adoption were scraped from www.petfinder.com.

# Repository Contents

This github repository contains the following files.

1. 01_web_scraping.ipynb - contains the main web scraping process used to gather the dog breed characteristics and dogs for adoptions datasets
2. 02_api_calls.ipynb - contains the API calls that were ran to download dog images from www.dog.ceo
3. 03_clean_split_preprocess_data.ipynb - notebook that has all of the initial data preparations prior to EDA
4. 04_EDA.ipynb - contains deep-dive analysis of the images dataset through visualizations and observations
5. 05_image_classification.ipynb - contains the actual step-by-step process in building multi-class image classification models

# Data Understanding

## Class Distribution

Below is the class disstribution of 173 different dog breeds.

![](figures/classdist.png)

There are a total of 173 breeds making this a complex multi-class image classification. Furthermore, there is also a wide disparity in the amount of images available for each class, resulting to an obvious class imbalance. 

There are various ways that I could deal with the class imbalance. I am predicting that our model to be build from scratch will perform very poorly given the lack of observations for half of the breeds. However, I am confident that using pre-trained models via Transfer Learning can help solve this problem. We will compensate the lack of images for some of the breeds by tapping into the other images that the pre-trained models have trained on.

## Peeking at the Images

Take a look at some of the sample images.

![](figures/sampleimages.png)

Given the wide variety of dogs as seen in the random images, this proves that there is indeed a need for a dog breed classifier. Normal folks would find it hard to identify a dog's breed because of the significant physical differences and the actual number of breeds there are.

## Image Sizes and Dimensions

Upon initial check of file sizes and dimensions, there were several outliers. The figures below zoom in on the majority of the files:

![](figures/size2.png)

![](figures/dimen2.png)

There are definite outliers when looking into the file size and dimensions of the images. Image quality does matter when tackling image classification but in order to avoid model biases and in the interest of efficiency, all images will be resized to standard measurements. 

# Modeling Process

These were the steps undertaken in building the various image classification models.

1. Data Preparation - This includes train and validation splitting, converting into tensors, separating into batches.
2. Build Baseline Model - This uses Dummy Classifier to build a baseline model based on the most frequent class. This model will not generate any insight about the data.
3. Build CNN Model from Scratch - Build the model from scratch with the goal of beating the baseline model score.
4. Use Transfer Learning Models - Building a machine learning model and training it on lots from scratch can be expensive and time consuming. Transfer learning helps alleviate some of these by taking what another model has learned and using that information with your own problem.
5. Model Evaluation and Selection - Evaluating the best model through different metrics
6. Making Predictions - Seeing the models in action, by predicting on validation images and custom, user-provided images

# Metrics and Evaluation

Here are the metrics trend of the best performing pre-trained model.

![](figures/bestmodel.png)

![](figures/bestmodel2.png)

In each type of modeling, I am using loss as the basis of which epoch is the best. But in models-to-models comparison, I am using accuracy as my final metric because I am not handling sensitive data like medical decisions where the balance between precision and recall matters. I will simply look at accuracy to check whether the model has correctly predicted a breed. 

The final model has an accuracy of 88%, which is very good considering the lack of images of half of the breeds and the number of classes.

![](figures/conf.png)

# Predictions

## Predicting Validation Data

1. Predicting invidual image

![](figures/pred1.png)

2. Showing all prediction probabilities

![](figures/pred2.png)

3. Sample set of predictions

![](figures/pred3.png)

## Predicting Custom Images

![](figures/custom.png)

## False Predictions

Looking closely at the false predictions, the primary reason they are failing is because of uncanny physical and appearance similarities between two of the breeds. 

1. Sample number one is when they appear similar as in the case of a Labrador Retriever versus a Flatcoated Retriever.

Labrador

![](figures/labrador.jpg)

Flat-coated Retriever

![](figures/flatcoat.jpg)


2. Another instance is when both breeds have same unique features or markings such as irregular black splotches of Bluetick Coonhounds and English Setter.

Bluetick Coonhound

![](figures/bluetick.jpg)

English Setter

![](figures/setter.jpg)

# Further Improvements

This is not yet the complete version of the project. I expect to attach an actual recommendation system into this image classification model that would recommend specific dog breeds to individuals based on the lifestyle. 

But in terms of further improvements specific to my dog breed image classification:
1. Try other pre-trained models (e.g. VGG16, VGG19)
2. Add data augmentation which takes the training images and manipulates (crop, resize) or distorts them (flip, rotate) to create even more training features for the model to learn from
3. More finetuning using more and more dog images dataset