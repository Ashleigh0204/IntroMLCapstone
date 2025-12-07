# Intro ML Capstone Project
### Ashleigh SICO
### ITCS 5356-001
### 7 December 2025

Using machine learning to predict housing prices is a topic of great interest.  This report uses the Ames Housing dataset to train five machine learning models to predict the price of a house for sale in Ames, IA, USA.  A linear regression model, ridge regression model, and neural network regression model are implemented based on classical machine learning techniques.  Two literature-based machine learning model are also implemented for this dataset.  The literature-based models include a hybrid regression model proposed by Truong et al. using an random forest, XGBoost, and LightGBM model and a hybrid kernel extreme learning machine proposed by Yu et al.  The model performance is measures and compared for each of the chosen models.

`DataCleaning.py` contains the pipeline to clean the Ames Housing dataset by filling in missing values, one-hot encoding, and standarization.  `Regreesion.py` contains helper methods such as the cost function root mean squared error upon which the models rely.

The classical models are implemented in `Linear.py`, `Ridge.py`, and `NeuralNetwork.py`.  The literature-based models are implemented in `HybridRegression.py` and `HKELM.py`.  Each model will run using the training dataset and the validation dataset.  The output includes the RMSE for the training and validation datasets and a scatter plot comparing the predicted price to the actual price.

The dataset is included in the folder `house-price-advanced-regression-techniques`.