from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json 
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
values_X, values_Y = values[:, 1:], values[:, [0]]

#########################
### PROCESSING X DATA ###
#########################

'''
non iterable (String) to iterable (Integer) encoder
Encode only field index 4
then cast the encoded value to float
'''
encoder_X = LabelEncoder()
values_X[:,3] = encoder_X.fit_transform(values_X[:,3])
values_X = values_X.astype('float32')

# normalize features
scalerX = MinMaxScaler(feature_range = (0, 1))
scaledX = scalerX.fit_transform(values_X)

# normalize features
scalerY = MinMaxScaler(feature_range = (0, 1))
scaledY = scalerY.fit_transform(values_Y)

# split into train and test sets
n_train_hours = 365 * 24 * 2

# split into input and outputs
train_X, train_Y = scaledX[:n_train_hours, :], scaledY[:n_train_hours, :]
test_X, test_Y = scaledX[n_train_hours:, :], scaledY[n_train_hours:, :]

# print len(train_X), len(test_X)
# exit(0)

def transform_raw_data(np_arr_2d_data):
	test2_data = np_arr_2d_data
	pp0 = encoder_X.transform(test2_data[:,3])
	pp0 = pp0.astype('float32')

	len_data = len(test2_data)
	for i in range(len_data):
		test2_data[i][3] = pp0[i]
	pp1 = scalerX.transform(test2_data)
	return pp1

def transform_raw_data_Y(np_arr_2d_data):
	return scalerY.inverse_transform(np_arr_2d_data)

# print transform_raw_data(np.array([[-16,-4.0,1020.0,"SE",1.79,0,0]]))
# print transform_raw_data_Y(np.array([[0.12977867]]))
# exit(0)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
# exit(0)
 

def test():
	model_out_filename = 'test_model'
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
	scores = model.evaluate(test_X, test_Y, verbose=0)
	print "Accuracy: ", scores



	# 2014-09-07 21:00:00,129.0,20,24.0,1008.0,NW,1.79,0,0
	# 2014-09-07 22:00:00,140.0,20,22.0,1008.0,NW,2.68,0,0
	# 2014-09-07 23:00:00,156.0,20,23.0,1009.0,NW,5.81,0,0
	# 2014-09-08 00:00:00,104.0,19,21.0,1009.0,NW,8.94,0,0
	# 2014-09-08 01:00:00,2.0,16,20.0,1010.0,NW,13.86,0,0
	# 2014-09-08 02:00:00,2.0,10,19.0,1010.0,NW,16.99,0,0
	# 2014-09-08 03:00:00,6.0,10,17.0,1010.0,NW,20.12,0,0
	# 2014-09-08 04:00:00,5.0,7,18.0,1011.0,cv,0.89,0,0
	data_test = transform_raw_data(np.array([
		[20, 24.0, 1008.0, "NW",  1.79, 0, 0], # 129.0
		[20, 22.0, 1008.0, "NW",  2.68, 0, 0], # 140.0
		[20, 23.0, 1009.0, "NW",  5.81, 0, 0], # 156.0
		[19, 21.0, 1009.0, "NW",  8.94, 0, 0], # 104.0
		[16, 20.0, 1010.0, "NW", 13.86, 0, 0], #   2.0
		[10, 19.0, 1010.0, "NW", 16.99, 0, 0], #   2.0
		[10, 17.0, 1010.0, "NW", 20.12, 0, 0], #   6.0
		[ 7, 18.0, 1011.0, "cv",  0.89, 0, 0]  #.  5.0  
	]))
	inx = data_test.reshape((data_test.shape[0], 1, data_test.shape[1]))
	yhat = model.predict(inx)

	# print transform_raw_data_Y(np.array([[0.12977867]]))

	print transform_raw_data_Y(yhat)

# print train_X
test()
exit(0)
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_Y, epochs=50, batch_size=36, verbose=2, shuffle=False)

scores = model.evaluate(test_X, test_Y, verbose=0)
print "Accuracy: ", scores

# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# # make a prediction
# yhat = model.predict(test_X)
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

model_out_filename = 'test_model'

# serialize self.model to JSON
model_json = model.to_json()
with open("model/" + model_out_filename + ".json", "w") as json_file:
	json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model/" + model_out_filename + ".h5")
print("Model " + model_out_filename + " saved to disk")

