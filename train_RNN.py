# import required packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential, losses
import h5py
from pickle import dump
import os
warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


def create_dataset(data, past_days):
    new_label_list=[]
    
    for i in range(past_days):
        new_label_list.append("volume_"+str(i+1))
        new_label_list.append("open_"+str(i+1))
        new_label_list.append("high_"+str(i+1))
        new_label_list.append("low_"+str(i+1))
    new_label_list.append("target")
    new_data = pd.DataFrame(0,index=np.arange(len(data)-past_days),columns=new_label_list)
    return new_data

if __name__ == "__main__": 
	# 1. load your training data
 
 
	data = pd.read_csv("./q2_dataset.csv")
	data_reverse = data.reindex(index=data.index[::-1]).reset_index(drop=True)
	new_data=create_dataset(data,3)
	for i in range(len(new_data)):
		for j in range(3):
			new_data['volume_'+str(j+1)][i]=data_reverse[' Volume'][j+i]
			new_data['open_'+str(j+1)][i]=data_reverse[' Open'][j+i]
			new_data['high_'+str(j+1)][i]=data_reverse[' High'][j+i]
			new_data['low_'+str(j+1)][i]=data_reverse[' Low'][j+i]
		new_data['target'][i]=data_reverse[' Open'][i+3]
	x_data=new_data.iloc[:,:-1]
	y_data=new_data.iloc[:,-1]
 
	X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.30, random_state=42)
 
	train_data = pd.concat([X_train, y_train],  axis=1)
	test_data = pd.concat([X_test,y_test], axis=1)
 
	train_data.to_csv('./data/train_data_RNN.csv',index=False)  
	test_data.to_csv('./data/test_data_RNN.csv',index=False)
 
	train_data = pd.read_csv("./data/train_data_RNN.csv")
	
	x_train_data=train_data.iloc[:,:-1]
	y_train_data=train_data.iloc[:,-1]
	
	sc_features = MinMaxScaler()
	sc_label=MinMaxScaler()
	x_train_data=sc_features.fit_transform(x_train_data)
	y_train_data=sc_label.fit_transform(y_train_data.to_numpy().reshape(-1,1))
 
	a = x_train_data.reshape(879,12,1)
	b = y_train_data.reshape(-1,1,1)

 
	

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss
 
	# SimpleRnn=Sequential()
	# SimpleRnn.add(layers.SimpleRNN(units=50, input_shape=(a.shape[1],1), activation='relu',return_sequences=True))
	# SimpleRnn.add(layers.Dropout(0.2))
	# SimpleRnn.add(layers.SimpleRNN(units=30,return_sequences=True))
	# SimpleRnn.add(layers.Dropout(0.2))
	# SimpleRnn.add(layers.SimpleRNN(units=30,return_sequences=False))
	# SimpleRnn.add(layers.Dropout(0.2))
	# SimpleRnn.add(layers.Dense(units=1))
	# SimpleRnn.compile(loss='mean_squared_error', optimizer='adam')
 
	# SimpleRnn_history=SimpleRnn.fit(a,b , epochs=100, batch_size=32)
 
	# loss_simpleRnn=SimpleRnn_history.history['loss']
	# plt.plot(loss_simpleRnn,label='Train Loss')
	# plt.show()
 
	lstm_model=Sequential()
	lstm_model.add(layers.LSTM(units=70, input_shape=(a.shape[1],1), activation='relu',return_sequences=True))
	lstm_model.add(layers.Dropout(0.2))
	lstm_model.add(layers.LSTM(units=50,return_sequences=True))
	lstm_model.add(layers.Dropout(0.2))
	lstm_model.add(layers.LSTM(units=30,return_sequences=False))
	lstm_model.add(layers.Dropout(0.2))
	lstm_model.add(layers.Dense(units=1))


	lstm_model.compile(loss='mean_squared_error', optimizer='adam')
	lstm_model.summary()
 
 

	lstm_history=lstm_model.fit(a,b , epochs=150, batch_size=32)
 
	lstm_loss=lstm_history.history['loss']
 
	for epoch, loss in enumerate(lstm_loss, 1):
		print(f"Epoch [{epoch}/{len(lstm_loss)}], Loss: {loss:.4f}")
  
	total_training_loss = sum(lstm_loss)
	num_epochs = len(lstm_loss)
	avg_training_loss = total_training_loss / num_epochs
	print(f"Final Training Loss: {avg_training_loss:.4f}")
	plt.plot(lstm_loss,label='Train Loss')
	plt.show()
 
	# gru_model=Sequential()
	# gru_model.add(layers.GRU(units=70, input_shape=(a.shape[1],1), activation='relu',return_sequences=True))
	# gru_model.add(layers.Dropout(0.2))
	# gru_model.add(layers.GRU(units=50,return_sequences=True))
	# gru_model.add(layers.Dropout(0.2))
	# gru_model.add(layers.GRU(units=30,return_sequences=False))
	# gru_model.add(layers.Dropout(0.2))
	# gru_model.add(layers.Dense(units=1))

	# gru_model.compile(loss='mean_squared_error', optimizer='adam')
 
	# gru_model_history=gru_model.fit(a,b , epochs=150, batch_size=32)
 
	# gru_loss=gru_model_history.history['loss']
	# plt.plot(gru_loss,label='Train Loss')
	# plt.show()
 
 
	# 3. Save your model
	file_path_data = 'data/minMax_data.pkl'
	file_path_label = 'data/minMax_label.pkl'
	lstm_model.save("model/GROUP_26_RNN_model.h5")
	with open(file_path_data, 'wb') as f:
		dump(sc_features, f)
	with open(file_path_label, 'wb') as f:
		dump(sc_label, f)
	