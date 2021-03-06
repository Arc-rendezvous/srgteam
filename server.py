import __future__
from Forecaster import Forecaster
import os
from flask import Flask, request, json, render_template, Response
from flask_cors import CORS
import json
import datetime


app = Flask(__name__)
CORS(app)
	
def pretty(d, indent=0):
	s = ""
	for key, value in d.items():
		s += '\t' * indent + str(key) + "\n"
		if isinstance(value, dict):
			pretty(value, indent+1)
		else:
			s += '\t' * (indent+1) + str(value) + "\n"

	return s + "\n"

@app.route('/')
def hello_world():
	return 'Hello World'

@app.route('/predict', methods = ['GET'])
def predict():
	cons = request.args.get('cons', default = 'gasoline', type = str)
	predict = request.args.get('predict', default = 1, type = int)
	step = request.args.get('step', default = 0, type = int)

	if not cons[0] in ['g', 'd']:
		return "cons param error"

	if predict < 1 or predict > 50:
		return "predict param error"

	data_fn = 'data/data-month-final.csv'
	shift = 50
	kind = cons[0]
	fc = Forecaster(data_fn, shift, 5)

	arr = [x[0] for x in fc.predict_out('test_model_lstm_6_1_shift_' + str(shift) + '_almost_full_data_' + str(kind) + '__')]
	arr = [float(x) for x in arr]
	arr2 = [x[0] for x in fc.scaler_Y.inverse_transform(fc.Ytrain)]

	concat_index = len(arr2)
	middle_index = concat_index + step
	cummulative_data = arr2 + arr

	start_index = middle_index - 7
	end_index = middle_index + 7

	out_current = []
	out_pred = []

	if start_index >= concat_index:
		out_pred = cummulative_data[start_index:end_index]
	elif end_index < concat_index:
		out_current = cummulative_data[-14:]
	else:
		out_current = cummulative_data[start_index:concat_index]
		out_pred = cummulative_data[concat_index:end_index]

	now = datetime.datetime.now()
	# time = 

	date_now = now

	date_out_pred = []
	date_out_current = []

	len_left = len(out_current)
	len_right = len(out_pred)

	for i in range(step + 1 + len_left, step + 1 + len_left + len_right):
		date_x = date_now + datetime.timedelta(days=i - 8)
		date_out_pred.append(date_x.strftime("%Y-%m-%d"))

	for i in range(step + 1, step + 1 + len_left):
		date_x = date_now + datetime.timedelta(days=i - 8)
		date_out_current.append(date_x.strftime("%Y-%m-%d"))
	
	# date_out_current = []
	# for i in range(-step, -step)

	resp = Response(json.dumps({
		'past': out_current,
		'predict': out_pred,
		'date_predict': date_out_pred,
		'date_current': date_out_current
	}))
	resp.headers['Content-Type'] = 'application/json'
	return resp

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5003)
