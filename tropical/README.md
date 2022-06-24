# Pre-requisite to run the notebook
You need to register with Radiant ML Hub to get an API which will be used to access the data in the notebook. To get your API key, go to dashboard.mlhub.earth. If you have not used Radiant MLHub before, you will need to sign up and create a new account. Else, sign in. In the **API Keys** tab, you will be able to create API key(s), which you will need. _Do not share_ your API key with others: your usage may be limited and sharing your API key is a secuirty risk.

# Tropical Cyclone Use Case

## Problem Motivation

Tropical cyclones cause billions of dollars of damages and kill hundreds of people globally every year.  High winds are a leading cause of damage both via direct damage to buildings and infrastructure, and indirectly by driving storm surges.  While great improvements have been made in predicting the <b>track</b>, or location, of tropical cyclones in recent years, tropical cyclone <b>intensity</b> (maximum wind speed) is still challenging to forecast, and forecasting centers such as the National Hurricane Center in the United States have not been able to make substantial improvements in intensity forecasts, even as computing power has improved (see Figures 1-2 in [Cangialosi et al. 2020](https://www.nhc.noaa.gov/pdf/Cangialosi_et_al_2020.pdf)).  More recently, scientists have turned to deep learning-based models of tropical cyclone wind speed as a possible avenue for improving intensity forecasts.  In 2018, the NASA IMPACT team launched an experimental framework to explore the possible applications of deep learning and tropical cyclone intensity forecasting.  One of their goals is to use satellite images of storms to improve wind forecasts.  

In previous decades, forecasters have estimated tropical cyclone intensity using the [Dvorak technique](https://en.wikipedia.org/wiki/Dvorak_technique), which requires each individual forecaster to identify patterns in tropical storms based on visual identification of cloud features in visible and infrared satellite imagery.  As you can imagine, this is pretty labor-intensive, and subjective to the interpretation of each individual forecaster! The hope is that an accurate machine learning-based model would be faster, automated, and more objective than relying on human forecasters.   

## Goals
For this summer school, we have pre-fit ML models (logistic regression, random forest, and gradient-boosted trees) for both the classification and regression task. Our primary goal is to become familiar with the dataset/ML models and assess what the ML models have learned. In short our goals are the following:
1. Become familiar with the model performance 
2. Determine the important features 
3. Determine the learned relationships 
4. Assess explanations for individual predictions

By gaining a deeper understanding of the dataset/model, we can start to assess the trustwortiness of the model. We have provided template notebooks with examples of model explainability for a single model for the classification and regression tasks. In each of them there are series of question for the participants to ponder and discuss as it pertains to model development, understanding, and implementation. We include optional tasks and the flexibility to evaluate different ML models. We encourage the participants to attempt these different tasks to ensure their understanding of the material. 

These notebooks expect that you complete the classification tasks before the regression tasks.

## Data and ML Models 

[The NASA Tropical Storm Competition dataset](https://mlhub.earth/data/nasa_tropical_storm_competition) leverages satellite imagery to create deep learning -based models of tropical cyclone wind speed.  The dataset consists of single-band satellite images and wind speed annotations from over 600 storms that have been labeled by NASA and the Radiant Earth Foundation.  The images are captured at different times in the storms' life cycles, and labeled with relative time since the beginning of the storm--this means that you can use past images of a given storm to inform your model, but you cannot use images from later on in a storm to predict winds earlier in the storm.

## Resources

[An Evaluation of Dvorak Technique–Based Tropical Cyclone Intensity Estimates](https://journals.ametsoc.org/view/journals/wefo/25/5/2010waf2222375_1.xml)
