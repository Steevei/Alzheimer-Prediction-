# Alzheimer's Disease Prediction using MRI Data

This repository contains a Jupyter Notebook for predicting age groups of individuals based on MRI data, specifically to aid in understanding Alzheimer's disease progression. The data is sourced from a Kaggle dataset and includes various features relevant to the study.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
This project aims to predict the age group of individuals using MRI data with a focus on Alzheimer's disease. The models used for this task are:
- Decision Tree Classifier
- Logistic Regression
- Random Forest Classifier

## Dataset
The dataset used for this project is sourced from Kaggle. It contains MRI data and other relevant parameters for individuals of different age groups. You can find the dataset here: [MRI and Alzheimer's Dataset on Kaggle](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers/data?select=oasis_cross-sectional.csv).

The dataset includes features such as:
- ID
- Gender
- Age
- Handedness
- Education
- SES (Socio-Economic Status)
- MMSE (Mini-Mental State Examination)
- CDR (Clinical Dementia Rating)
- eTIV (Estimated Total Intracranial Volume)
- nWBV (Normalized Whole Brain Volume)
- ASF (Atlas Scaling Factor)

## Models Used
We are using three different models to predict the age group:
1. **Decision Tree Classifier**: A non-parametric supervised learning method used for classification and regression.
2. **Logistic Regression**: A statistical model that in its basic form uses a logistic function to model a binary dependent variable.
3. **Random Forest Classifier**: An ensemble learning method for classification that operates by constructing a multitude of decision trees during training.

## Acknowledgements
- The dataset is provided by [Kaggle](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers/data?select=oasis_cross-sectional.csv).
- The rest of the information that was used for reference can be found in the following links below:
  
- https://medium.com/analytics-vidhya/logistic-regression-in-python-using-pandas-and-seaborn-for-beginners-in-ml-64eaf0f208d2
- https://www.youtube.com/watch?v=29ZQ3TDGgRQ&list=LL&index=2&t=1666s
- https://www.youtube.com/watch?v=zM4VZR0px8E
- https://www.ibm.com/topics/random-forest
- https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
- https://scikit-learn.org/stable/modules/tree.html
- https://www.youtube.com/watch?v=PHxYNGo8NcI
- https://statisticsbyjim.com/basics/outliers/
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validatehttps://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
- https://towardsdatascience.com/decision-tree-classifier-explained-in-real-life-picking-a-vacation-destination-6226b2b60575
- https://note.nkmk.me/en/python-pandas-value-counts/
- https://scikit-learn.org/stable/modules/preprocessing.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- 
---

*Stefan Petrovski, University of Primorska UP-FAMNIT, Bioinformatics, 2024*
