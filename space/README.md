# Space Weather Use Case - Modeling the geomagnetic Disturbance Storm Time (Dst) Index

## Problem Motivation
The efficient transfer of energy from solar wind into the Earth’s magnetic field causes geomagnetic storms. The resulting variations in the magnetic field increase errors in magnetic navigation. The disturbance-storm-time index, or Dst, is a measure of the severity of the geomagnetic storm.

As a key specification of the magnetospheric dynamics, the Dst index is used to drive geomagnetic disturbance models such as NOAA/NCEI’s [High Definition Geomagnetic Model - Real-Time](https://www.ngdc.noaa.gov/geomag/HDGM/hdgm_rt.html) (HDGM-RT).
![HDGMRT_GRAPHIC_URL](https://www.ngdc.noaa.gov/geomag/HDGM/images/HDGM-RT_2003_storm_720p.gif "HDGM-RT")
 
In 2020-2021, NOAA and NASA conducted an international crowd sourced data science competition “MagNet: Model the Geomagnetic Field”:
https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/

Empirical models have been proposed as early as in 1975 to forecast DST solely from solar-wind observations at the Lagrangian (L1) position by satellites such as NOAA’s Deep Space Climate Observatory (DSCOVR) or NASA's Advanced Composition Explorer (ACE). Over the past three decades, several models were proposed for solar wind forecasting of DST, including empirical, physics-based, and machine learning approaches. While the ML models generally perform better than models based on the other approaches, there is still room to improve, especially when predicting extreme events. More importantly, we intentionally sought solutions that work on the raw, real-time data streams and are agnostic to sensor malfunctions and noise.
 
Thus, the competition task was to develop models for forecasting Dst that push the boundary of predictive performance, under operationally viable constraints, using the real-time solar-wind (RTSW) data feeds from NOAA’s DSCOVR and NASA’s ACE satellites. Improved models can provide more advanced warning of geomagnetic storms and reduce errors in magnetic navigation systems. Specifically, given one week of data ending at t minus 1 minute, the model must forecast Dst at time t and t plus one hour. As an attendee of the 2022 Trustworthy Artificial Intelligence for Environmental Science ([TAI4ES](https://www2.cisl.ucar.edu/events/tai4es-2022-summer-school)) Summer School, you will explore the benchmark and one of the winning models to learn more about modeling Dst. Magnetic surveyors, government agencies, academic institutions, satellite operators, and power grid operators use the Dst index to analyze the strength and duration of geomagnetic storms. As you’ve learned in the summer school lecture series, users are an essential part of developing trustworthy AI. For the trust-a-thon you were assigned a user to consider throughout the activities of the trust-a-thon. You will be assigned one of two  personas:  The “Precision Navigator” exploring without GPS or the “Flight Planner” concerned with space weather situational awareness for radio communications availability at high latitudes. When the discussion questions refer to “your user,” you should think about the persona you were assigned.  
 

## Goals

### Overall

The overall goal is to learn about ML modeling a key space weather storm indicator Dst.

We will use two notebooks through the summer school experience:
1. magnet_lstm_tutorial.ipynb - ideal for students newer to AI/ML.
2. magnet_cnn_tutorial.ipynb - ideal for students with more AI/ML experience.


### Day 1 - Space Weather 101: Solar Wind and Geomagnetic Storms

Goal: Be able to run both MagNet notebooks successfully, and also explore relationships between the solar wind input data “features” and <i>Dst</i> predictions (“labels”), as pre-modeling XAI.

Discussion Questions:
1. Can you describe the physical process between solar wind and ground geomagnetic disturbances? What is the Dst index primarily used for? [Hint: Large changes in solar wind velocity and density combined with the magnetic field oriented southward typically results in significant changes in the geomagnetic field near the Earth’s surface about an hour later.]
2. Roughly 85% of the time, near Earth is geomagnetically quiet. How might these infrequent solar wind events make modeling their predicted effects challenging? How might you make an accurate model with very few extreme events/samples?
3. How are the input data differently distributed? How are the input data correlated or uncorrelated with each other? How are they correlated with Dst? 
4. Based on what you’ve learned so far, what do you think your user will care about most? How do you think you can help your user view this model as trustworthy? 

### Day 2 - Deeper XAI

Goal: Understand tradeoffs between our simple LSTM and more complex CNN model.

Discussion Questions:
1. How well do the CNN and LSTM models perform? 
2. Do you expect them to perform better or worse for quiet or active space weather?
3. What are the model sensitivities to the input parameters? If one or more solar wind instruments were to degrade on orbit how do you predict that might impact model performance? 
4. Which model do you think best fits your end user’s needs? How would you communicate your decision and the associated tradeoffs to your end user?

### Day 3 - Data Quality

Goal: Review case studies, explore data degradation, and contemplate the value in incorporating new future training data.

Discussion Questions:
1. Thinking about your predictions from yesterday, if you degrade the data for one or more input parameters by adding Gaussian noise, how does that impact the model performance?
2. Based on what you learned in the lectures today and your review of these case studies, what would you show your end user if they asked for a case study? How would you do this? 

### Day 4 - Open Exploration

Goal: Open dialog and student driven space weather exploration.

Discussion Questions:
1. Imagine we experience a super-geomagnetic storm like that of 1859, the “Carrington Event” (e.g. see [Tsuritani et al., 2003](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2002JA009504)). How might we prepare our model to predict rare and such large events? 
2. Given your experience this week, what Trustworthy and Explainable AI suggestions would you like to make for modeling Dst and other space weather indicators? How would you communicate this to your end user? 

## Data
Data used for this challenge are publicly available from NOAA at these links:
* https://ngdc.noaa.gov/geomag/data/geomag/magnet/public.zip
* https://ngdc.noaa.gov/geomag/data/geomag/magnet/private.zip

Note that the competition discussed above used public data for development and the public leaderboard. A private dataset was kept internal during the competition for use in scoring by organizers. Since the competition has passed, both datasets are publicly accessible from NOAA. We will build and evaulate the models using the competition's public data and evaluate storm event case studies using the competition's "private" data.

## ML Models

### Ensemble Convolutional Neural Network (CNN)  model with Relu activation
This was the 2nd place winning model from the MagNet competition. The model is a convolutional neural network with an architecture designed to give more importance to later points of the time series, while also capturing larger-scale patterns over the whole series. The network consists of a set of convolutional layers which detect patterns at progressively longer time spans. Following all the convolutional layers is a layer which concatenates the last data point of each of the convolution outputs. This concatenation is then fed into a dense layer. The idea of taking the last data point of each convolution is that it represents the patterns  at different time spans leading up to the prediction time: for example, the last data point of the first layer gives the features of the hour before the prediction time, then the second layer gives the last 6 hours, etc.

The architecture is somewhat similar to a widely used architecture for image segmentation, the U-Net introduced by Ronneberger et al., (2015). The U-Net consists of a "contracting path", a series of convolutional layers which condense the image, followed by an "expansive path" of up-convolution layers which expand the outputs back to the scale of the original image. Combining small-scale and large-scale features allows the network to make localized predictions that also take account of larger surrounding patterns. The idea is also similar to the Temporal Convolutional Network described by Bai et al., 2018; however their architecture uses residual (i.e. additive) connections to blend the low-level and high-level features, rather than concatenations.

Missing data is filled by linear interpolation (to reduce noise, the interpolation uses a smoothed rolling average, rather than just the 2 points immediately before and after the missing part). Features are normalized by subtracting the median and dividing by the interquartile range (this approach is used rather than the more usual mean and standard deviation because some variables have asymmetric distributions with long tails). Data is aggregated in 10-minute increments, taking the mean and standard deviation of each feature in the increment.

The final model is an ensemble of 5 models with the same structure, trained on different subsets of the data. Separate models are trained for times t and t + 1, yielding 10 models in total. This technique is often called "cross-validation folds", and is a common technique in machine learning. The idea is that each model only imperfectly captures the "true" relationship between the input and output variables, and partly fits to noise in the training data. But if we average several models, the random noise components will approximately cancel each other out, leaving a more accurate prediction of the true relationship.

### Long Short Term Memory (LSTM) Model

This is the benchmark model provided by the MagNet competition organizers. Long Short Term Memory networks or LSTMs are a special kind of recurrent neural network especially suited to time series data. In the related notebook, we will show you how to implement a first-pass LSTM model for predicting <i>Dst</i>. This notebook is ideally suited for students newer to AI/ML topics. In this notebook, you'll find many embedded prompts with useful tips, optional tasks and questions to ponder.

### Disclaimer

The United States Department of Commerce (DOC) GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. DOC has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information. Any claims against the Department of Commerce stemming from the use of its GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
