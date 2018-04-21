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
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json 
from keras.optimizers import SGD

from sklearn.metrics import accuracy_score
 
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

number_of_shifted = 2
df = DataFrame(scaled)
df_shifted = df.shift(number_of_shifted)

data_for_scaler_y = []
for arr_data in values:
	data_for_scaler_y.append([arr_data[0]])

scaler_Y = MinMaxScaler(feature_range=(0, 1))
scaled_Y = scaler_Y.fit_transform(data_for_scaler_y)

# print data_for_scaler_y[0]
# print [scaled_Y[0]]
# print scaler_Y.inverse_transform([scaled_Y[0]])
# exit()

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

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xforecast = np.array(Xforecast)

splitter_number = 365 * 24
# split into input and outputs
train_X, train_Y = Xtrain[:splitter_number, :], Ytrain[:splitter_number, :]
test_X, test_Y = Xtrain[splitter_number:, :], Ytrain[splitter_number:, :]
# print train_X[0]
# exit()
# reshape input to be 3D [samples, timesteps, features]
# print train_X.shape[0], train_X.shape[1]
# exit()
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# print train_X[0]
# datax = [x[0] for x in train_X]
# print datax[0]
# print list(scaler.inverse_transform([datax[0]]))
# exit()

def test():
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
	model.compile(loss='mae', optimizer='adam')
	print "Error is ", model.evaluate(test_X, test_Y) * 100, "%"

	tes_data = test_X[-20:]
	print "X", tes_data

	tes_datay = test_Y[-20:]
	print "Y", tes_datay

	# exit()
	# print Xforecast
	yhat = model.predict(tes_data)

	test_data_error = np.abs(tes_datay - yhat)
	actual_error = np.abs(scaler_Y.inverse_transform(tes_datay) - scaler_Y.inverse_transform(yhat))
	datax = [x[0] for x in tes_data]
	print list(scaler.inverse_transform(datax))
	print yhat
	print scaler_Y.inverse_transform(yhat)
	print test_data_error
	print actual_error

def predict_out():
	model_out_filename = 'test_model_lstm_6_1_shift_2'
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
	print "Error is ", model.evaluate(test_X, test_Y) * 100, "%"

	tes_data = Xforecast
	# exit()
	# print Xforecast
	yhat = model.predict(tes_data)

	test_data_error = np.abs(yhat - yhat)
	actual_error = np.abs(scaler_Y.inverse_transform(yhat) - scaler_Y.inverse_transform(yhat))
	datax = [x[0] for x in tes_data]
	print "input data: ", list(scaler.inverse_transform(datax))
	print yhat, "->", scaler_Y.inverse_transform(yhat)

def retrain():
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

# test()
# retrain()
predict_out()
exit()

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(units=30, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_Y, epochs=50, batch_size=72, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

model_out_filename = 'test_model_lstm_6_1_shift_2'

# serialize self.model to JSON
model_json = model.to_json()
with open("model/" + model_out_filename + ".json", "w") as json_file:
	json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model/" + model_out_filename + ".h5")
print("Model " + model_out_filename + " saved to disk")

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_Y = test_Y.reshape((len(test_Y), 1))
# inv_y = concatenate((test_Y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)

