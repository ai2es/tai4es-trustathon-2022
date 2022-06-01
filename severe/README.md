# Severe Use Case

## Problem Motivation

[Geostationary Lightning Mapper (GLM)](https://www.goes-r.gov/spacesegment/glm.html), which measures the presence of lightning, is available on Geostationary Environmental Satellite System (GOES)-16, but not previous GOESs. To create a long-term lightning climatology, we need a method for measuring the presence of lightning for GOES data prior to GOES-16 (i.e., November 2016). Thus, we are motivated to determine the following:
1. Does this storm-centered image contain a lightning flash? (Classification)
2. If so, how many lightning flashes? (Regression)


## Goals
For this summer school, we have pre-fit ML models (logistic regression, random forest, and gradient-boosted trees) for both the classification and regression task. Our primary goal is to become familiar with the dataset/ML models and assess what the ML models have learned. In short our goals are the following:
1. Become familiar with the model performance 
2. Determine the important features 
3. Determine the learned relationships 
4. Assess explanations for individual predictions

By gaining a deeper understanding of the dataset/model, we can start to assess the trustwortiness of the model. We have provided template notebooks with examples of model explainability for a single model for the classification and regression tasks. In each of them there are series of question for the participants to ponder and discuss as it pertains to model development, understanding, and implementation. We include optional tasks and the flexibility to evaluate different ML models. We encourage the participants to attempt these different tasks to ensure their understanding of the material. 

These notebooks expect that you complete the classification tasks before the regression tasks.


## Data and ML Models 

[The Storm EVent ImagRy (SEVIR) dataset](https://proceedings.neurips.cc/paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf) is a spatiotemporal dataset curated on weather events. The SEVIR dataset is based on 3 channels (C02, C09, C13) from Geostationary Environmetnal Satellite System (GOES)-16, Next-Generation Radar (NEXRAD) vertically integrated liquid, and GOES-16 Geostationary Lightning Mapper (GLM) flashes. Using these weather-centered imagery, Chase et al. extracted spatial percentiles
from the visible reflectance, water vapor brightness temperature, infrared, and VIL. Additional details on the feature engineering can be found at this [link](https://github.com/ai2es/WAF_ML_Tutorial_Part1/blob/main/jupyter_notebooks/Notebook02_Feature_Engineering.ipynb). For this summer school program, we have coarsened the resolution of the SEVIR dataset to faciliate comparsion between traditional and deep learning techniques. We have cleverly re-named this dataset as sub-SEVIR. 

## Notes
* [Randy Chase's SEVIR Colab Notebook](https://colab.research.google.com/drive/1pxJo458Ol0uLcAPyQWHyJpldze-tjnqG?usp=sharing)
* [WAF Tutorials on Machine Learning for Operational Meterology](https://github.com/ai2es/WAF_ML_Tutorial_Part1)