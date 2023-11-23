# Stock-Price-Prediction-using-RNN

## Introduction:
This repository focuses on predicting stock opening prices utilizing Recurrent Neural Networks (RNNs). The model is trained on historical stock data, and the project provides scripts for training and testing the RNN, along with an explanation of the dataset, model architecture, and performance evaluation.

## Files and Structure:
**train_RNN.py:** This script loads the dataset, creates a structured dataset for training, preprocesses the data, defines the RNN model architecture (LSTM), trains the model, and saves the trained model along with necessary preprocessing objects.
**Test_RNN.py:** This script loads the trained model and preprocessing objects, evaluates the model on test data, and visualizes the predicted vs. true values. It also calculates and prints the Root Mean Squared Error (RMSE) as a measure of model performance.

## Dataset:
* The dataset, q2_dataset.csv, spans five years with daily samples of Open, High, Low prices, and volume.
* Data preprocessing involves reversing the dataset for easier processing and creating a structured dataset with features for the past three days and the next day's opening price as the target variable.

## Training:
* The model is trained for 150 epochs with a batch size of 32 using Mean Squared Error loss and the Adam optimizer.
* Training progress is visualized with a plot of the training loss for each epoch.
  
## Testing and Evaluation:
* The trained model is tested on a separate test dataset, and the RMSE is calculated to assess predictive performance.
* The script also visualizes the predicted vs. true values to provide a qualitative understanding of the model's accuracy.
  
## Feature Window Experimentation:
The project includes an analysis of the impact of varying the feature window size from 4 to 10 days on training and testing performance. Results highlight a trade-off between the feature window size and model performance.

## Dependencies:
Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow, h5py
