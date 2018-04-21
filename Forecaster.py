from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import model_from_json 
from keras.optimizers import SGD

from sklearn.metrics import accuracy_score


class Forecaster:
	def __init__(self, data_filename, number_shifted_prediction, index_prediction):
		# load dataset
		self.dataset = read_csv(data_filename,
			sep=",", 
			header=0, 
			usecols=["stocktotal",
				"stockgasoline", 
				"stockdistilled",
				"pricegasoline", 
				"pricedistilled", 
				"consgasoline", 
				"consdistilled", 
				"gdp",
				"DPI",
				"totalsalesvehicle",
				"customerpriceindex",
				"vmt"])
		self.number_shifted_prediction = number_shifted_prediction
		self.index_prediction = index_prediction
		self.read_and_split_dataset()

	def read_and_split_dataset(self):
		values = self.dataset.values

		# encode non interable object to iterable (integer)
		# encoder = LabelEncoder()
		# values[:,4] = encoder.fit_transform(values[:,4])

		# ensure all data is float
		values = values.astype('float32')

		# normalize features to range [0,1]
		self.scaler = MinMaxScaler(feature_range=(0, 1))
		scaled = self.scaler.fit_transform(values)

		# how many data to be predicted
		number_of_shifted = self.number_shifted_prediction

		df = DataFrame(scaled)
		df_shifted = df.shift(number_of_shifted)

		data_for_scaler_y = []
		for arr_data in values:
			data_for_scaler_y.append([arr_data[self.index_prediction]])

		self.scaler_Y = MinMaxScaler(feature_range=(0, 1))
		scaled_Y = self.scaler_Y.fit_transform(data_for_scaler_y)

		# DATA FOR TRAIN & VALIDATION
		number_of_data_train_validation = len(df.values) - number_of_shifted
		number_of_data_forecasting = number_of_shifted
		Xtrain, Ytrain = [], []
		for i in range(number_of_data_train_validation):
			Xtrain.append(df.values[i])
			Ytrain.append(scaled_Y[i + number_of_shifted])

		Xforecast = []
		offset_index_data_forecasting = number_of_data_train_validation
		for i in range(number_of_shifted):
			index = offset_index_data_forecasting + i
			Xforecast.append([df.values[index]])

		self.Xtrain = np.array(Xtrain)
		self.Ytrain = np.array(Ytrain)
		self.Xforecast = np.array(Xforecast)

		# how many data to split train and test/validation
		splitter_number = int(0.8 * len(self.Xtrain))

		# split into input and outputs
		train_X, self.train_Y = self.Xtrain[:splitter_number, :], self.Ytrain[:splitter_number, :]
		test_X, self.test_Y = self.Xtrain[splitter_number:, :], self.Ytrain[splitter_number:, :]
		self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
		self.test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


	def test(self, model_filename, dataX, dataY):
		model_out_filename = model_filename
		'''
		LOADED MODEL BEGIN
		'''
		# load json and create model
		json_file = open('model/' + model_out_filename + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		
		# load weights into new self.model
		model.load_weights("model/" + model_out_filename + ".h5")
		print("Loaded model from disk")
		model.compile(loss='mae', optimizer='adam')
		print "Error is ", model.evaluate(self.test_X, self.test_Y) * 100, "%"

		tes_data = dataX
		print "X", tes_data

		tes_datay = dataY
		print "Y", tes_datay

		# exit()
		# print Xforecast
		yhat = model.predict(tes_data)

		test_data_error = np.abs(tes_datay - yhat)
		actual_error = np.abs(self.scaler_Y.inverse_transform(tes_datay) - self.scaler_Y.inverse_transform(yhat))
		datax = [x[0] for x in tes_data]
		print list(self.scaler.inverse_transform(datax))
		print yhat
		print self.scaler_Y.inverse_transform(yhat)
		print test_data_error
		print actual_error

	def predict_out(self, model_filename):
		model_out_filename = model_filename
		'''
		LOADED MODEL BEGIN
		'''
		# load json and create model
		json_file = open('model/' + model_out_filename + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		
		# load weights into new self.model
		model.load_weights("model/" + model_out_filename + ".h5")
		print("Loaded model from disk")
		model.compile(loss='mae', optimizer='adam')
		print "Error is ", model.evaluate(self.test_X, self.test_Y) * 100, "%"

		tes_data = self.Xforecast
		# exit()
		# print Xforecast
		yhat = model.predict(tes_data)

		test_data_error = np.abs(yhat - yhat)
		actual_error = np.abs(self.scaler_Y.inverse_transform(yhat) - self.scaler_Y.inverse_transform(yhat))
		datax = [x[0] for x in tes_data]
		# print "input data: ", list(self.scaler.inverse_transform(datax))
		# print yhat, "->", self.scaler_Y.inverse_transform(yhat)

		return self.scaler_Y.inverse_transform(yhat)

	def retrain(self):
		model_out_filename = 'test_model_lstm_6_1'
		'''
		LOADED MODEL BEGIN
		'''
		# load json and create model
		json_file = open('model/' + model_out_filename + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		
		# load weights into new self.model
		model.load_weights("model/" + model_out_filename + ".h5")
		print("Loaded model from disk")
		sgd = SGD(lr=0.001)
		model.compile(loss='mae', optimizer=sgd)

		# fit network
		history = model.fit(train_X, train_Y, epochs=50, batch_size=72, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

		model_out_filename = 'test_model_lstm_6_1'

		# # serialize self.model to JSON
		# model_json = model.to_json()
		# with open("model/" + model_out_filename + ".json", "w") as json_file:
		# 	json_file.write(model_json)

		# # serialize weights to HDF5
		# model.save_weights("model/" + model_out_filename + ".h5")
		# print("Model " + model_out_filename + " saved to disk")

		# plot history
		pyplot.plot(history.history['loss'], label='train')
		pyplot.plot(history.history['val_loss'], label='test')
		pyplot.legend()
		pyplot.show()

	def train(self, model_filename):

		# design network
		model = Sequential()
		model.add(LSTM(
			input_dim=12,
			output_dim=50,
			return_sequences=True))
		model.add(Dropout(0.2))

		model.add(LSTM(
			100,
			return_sequences=False))
		model.add(Dropout(0.2))

		model.add(Dense(
			output_dim=1))
		model.add(Activation('sigmoid'))
		model.compile(loss='mse', optimizer='adam')

		# fit network
		history = model.fit(self.train_X, self.train_Y, epochs=200, batch_size=4, validation_data=(self.test_X, self.test_Y), verbose=2, shuffle=False)

		model_out_filename = model_filename

		# serialize self.model to JSON
		model_json = model.to_json()
		with open("model/" + model_out_filename + ".json", "w") as json_file:
			json_file.write(model_json)

		# serialize weights to HDF5
		model.save_weights("model/" + model_out_filename + ".h5")
		print("Model " + model_out_filename + " saved to disk")

		# plot history
		# pyplot.plot(history.history['loss'], label='train')
		# pyplot.plot(history.history['val_loss'], label='test')
		# pyplot.legend()
		# pyplot.show()


data_fn = 'data/data-month-final.csv'

for i in range(15):
	shift = i + 1
	fc = Forecaster(data_fn, shift, 5)
	fc.train('test_model_lstm_6_1_shift_' + str(shift) + '_yay__')
	print fc.predict_out('test_model_lstm_6_1_shift_' + str(shift) + '_yay__')

for i in range(15):
	shift = i + 1
	fc = Forecaster(data_fn, shift, 6)
	fc.train('test_model_lstm_6_1_shift_' + str(shift) + '_yaz__')
	print fc.predict_out('test_model_lstm_6_1_shift_' + str(shift) + '_yaz__')
# fc.test('test_model_lstm_6_1_shift_1_yaz', fc.test_X[-10:], fc.test_Y[-10:])
