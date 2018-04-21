from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.metrics import accuracy_score

class SVRRegression:
	def __init__(self, data_filename, number_shifted_prediction, index_prediction):
		# load dataset
		self.dataset = read_csv(data_filename,
			sep=",", 
			header=0, 
			usecols=["Date", "Year", 
				"stocktotal",
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

		self.svr_rbf = svm.SVR(kernel='rbf', C=1e4, gamma=0.1)
		self.svr_lin = svm.SVR(kernel='linear', C=1e4)
		self.svr_poly = svm.SVR(kernel='poly', C=1e4, degree=4)

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
			Ytrain.append(scaled_Y[i + number_of_shifted][0])

		Xforecast = []
		offset_index_data_forecasting = number_of_data_train_validation
		for i in range(number_of_shifted):
			index = offset_index_data_forecasting + i
			Xforecast.append(df.values[index])

		self.Xtrain = np.array(Xtrain)
		self.Ytrain = np.array(Ytrain)
		self.Xforecast = np.array(Xforecast)

		# how many data to split train and test/validation
		splitter_number = int(0.8 * len(self.Xtrain))

		# split into input and outputs
		self.train_X, self.train_Y = self.Xtrain[:splitter_number, :], self.Ytrain[:splitter_number]
		self.test_X, self.test_Y = self.Xtrain[splitter_number:, :], self.Ytrain[splitter_number:]

	def fit_data(self):
		rbf_model = self.svr_rbf.fit(self.train_X, self.train_Y)
		y_rbf = rbf_model.predict(self.test_X)
		y_lin = self.svr_lin.fit(self.train_X, self.train_Y).predict(self.test_X)
		y_poly = self.svr_poly.fit(self.train_X, self.train_Y).predict(self.test_X)

		print "[SVR-rbf] Accuracy is ", mean_squared_error(self.test_Y, y_rbf)
		print "[SVR-lin] Accuracy is ", mean_squared_error(self.test_Y, y_lin)
		print "[SVR-pol] Accuracy is ", mean_squared_error(self.test_Y, y_poly)


		pred_xf = self.scaler_Y.inverse_transform([[x] for x in rbf_model.predict(self.Xforecast)])
		pred_as = self.scaler_Y.inverse_transform([[x] for x in self.test_Y])

		print pred_xf
		print pred_as

		pyplot.plot(pred_xf, label='rbf')
		# mix_data = self.test_Y + self.train_Y
		pyplot.plot(pred_as, label='asli')
		pyplot.legend()
		pyplot.show()

shift = 150
data_fn = 'data/data-month-final.csv'
fc = SVRRegression(data_fn, shift, 7)
fc.fit_data()
# print fc.predict_out('test_model_svr_6_1_shift_' + str(shift) + '_yay__')