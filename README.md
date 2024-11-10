# Charity Funding Success Prediction

This project is aimed at predicting the likelihood of success for charity funding applications using a neural network model. The dataset, `charity_data.csv`, contains information on various charity applications, such as application type, organization affiliation, income amount, and ask amount. This README outlines the setup, processing steps, and model creation process.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Overview
This project leverages machine learning techniques to predict whether a charity funding application will be successful. The solution involves preprocessing categorical data, scaling numerical features, and creating a neural network model with TensorFlow and Keras.

## Dataset
The dataset, `charity_data.csv`, includes columns such as:
- `EIN`: Employer Identification Number
- `NAME`: Charity name
- `APPLICATION_TYPE`: Type of application
- `AFFILIATION`: Affiliation type of the charity
- `CLASSIFICATION`: Classification code
- `USE_CASE`: Use case of the charity
- `ORGANIZATION`: Organization type
- `STATUS`: Application status
- `INCOME_AMT`: Income range of the charity
- `SPECIAL_CONSIDERATIONS`: If the charity has special considerations
- `ASK_AMT`: Amount requested
- `IS_SUCCESSFUL`: Target variable indicating if the funding was successful

## Data Preprocessing
1. **Dropping Unnecessary Columns**: The `EIN` column was dropped as it’s a unique identifier that doesn't contribute to model prediction.
   
2. **Reducing Dimensionality**:
   - **NAME**: Replaced unique charity names with "Other" for entries with fewer than five counts.
   - **APPLICATION_TYPE**: Grouped rare application types as "Other" based on a threshold of fewer than 500 entries.
   - **CLASSIFICATION**: Similar to `APPLICATION_TYPE`, classifications with fewer than 1000 entries were replaced with "Other."

3. **Encoding Categorical Variables**: Converted categorical columns to numerical representations using one-hot encoding.

4. **Feature Scaling**: Applied standard scaling to features to normalize the data.

## Model Architecture
The model is a neural network created using TensorFlow and Keras. It consists of:
- **Input Layer**: Matches the number of features in the scaled training set.
- **Hidden Layers**:
   - Layer 1: 8 nodes with ReLU activation.
   - Layer 2: 24 nodes with ReLU activation.
   - Layer 3: 42 nodes with ReLU activation.
- **Output Layer**: 1 node with sigmoid activation for binary classification (predicting success probability).

### Model Summary
```plaintext
Model: "sequential"
 Layer (type)           Output Shape       Param #  
 ─────────────────────────────────────────────────
 dense (Dense)          (None, 8)          3,576    
 ─────────────────────────────────────────────────
 dense_1 (Dense)        (None, 24)         216      
 ─────────────────────────────────────────────────
 dense_2 (Dense)        (None, 1)          25       
```
Total parameters: 3,817  
Trainable parameters: 3,817  

## Training and Evaluation
The model is compiled with the `binary_crossentropy` loss function, optimized using Adam, and evaluated with accuracy as a metric. It is trained over 100 epochs using the preprocessed and scaled data.

## Dependencies
This project requires:
- `pandas`: For data handling and preprocessing.
- `scikit-learn`: For data splitting and scaling.
- `tensorflow`: For building and training the neural network.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn tensorflow
   ```
2. **Run the Code**: Execute the script to preprocess data, build, and train the model:
   ```python
   python charity_funding_prediction.py
