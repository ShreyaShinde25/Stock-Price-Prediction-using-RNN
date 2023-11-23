# import required packages
from pickle import load
from tensorflow.keras.models import load_model
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def MSE(y_pred, y_true):
    diff=0
    rmse=0
    for i in range(len(y_pred)):
        diff+=((y_pred[i]-y_true[i])**2)
    rmse= np.sqrt(diff/len(y_pred))
    return rmse


if __name__ == "__main__":
	# 1. Load your saved model
	sc_features = load(open('data/minMax_data.pkl', 'rb'))
	sc_label = load(open('data/minMax_label.pkl', 'rb'))
	RNN_model = load_model('model/GROUP_26_RNN_model.h5')

	# 2. Load your testing data
	test_data = pd.read_csv("./data/test_data_RNN.csv")

	# 3. Run prediction on the test data and output required plot and loss
	x_test_data=test_data.iloc[:,:-1]
	y_test_data_org=test_data.iloc[:,-1]
	y_test_data=test_data.iloc[:,-1].to_numpy().reshape(-1,1)

	x_test_data=sc_features.transform(x_test_data)
	y_test_data=sc_label.transform(y_test_data)

	x_test_data = x_test_data.reshape(377,12,1)
	y_test_data = y_test_data.reshape(-1,1,1)
 
	# SimpleRnn_Score=SimpleRnn.evaluate(x_test_data,y_test_data, verbose=0)
	lstm_Score=RNN_model.evaluate(x_test_data,y_test_data, verbose=0)
	# gru_Score=gru_model.evaluate(x_test_data,y_test_data, verbose=0)

	# print("Loss for Simple RNN is",SimpleRnn_Score)
	print("Loss for LSTM is",lstm_Score)
	# print("Loss for GRU Model",gru_Score)
 
	# y_pred_simpleRnn=SimpleRnn.predict(x_test_data)
	y_pred_lstm=RNN_model.predict(x_test_data)
	# y_pred_gru=gru_model.predict(x_test_data)
	# y_pred_simpleRnn=sc_label.inverse_transform(y_pred_simpleRnn)
	y_pred_lstm=sc_label.inverse_transform(y_pred_lstm)
	# y_pred_gru=sc_label.inverse_transform(y_pred_gru)
 
	plt.figure(figsize=(10,5))
	plt.plot(y_pred_lstm,label="LSTM predicted value")
	plt.plot((y_test_data_org),label="True Value")
	plt.title("LSTM")
	plt.legend()
	plt.show()
 
	# SimpleRnn_rmse= MSE(y_pred_simpleRnn,y_test_data_org)
	LSTM_rmse=MSE(y_pred_lstm,y_test_data_org)
	# GRU_rmse=MSE(y_pred_gru,y_test_data_org)
	# print("Simple RNN RMSE",SimpleRnn_rmse)
	print("LSTM RMSE ",LSTM_rmse)
	# print("GRU RMSE",GRU_rmse)
 
