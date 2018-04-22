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
	shift = predict
	kind = cons[0]
	fc = Forecaster(data_fn, shift, 5)

	arr = [x[0] for x in fc.predict_out('test_model_lstm_6_1_shift_' + str(shift) + '_almost_full_data_' + str(kind) + '__')]
	arr = [float(x) for x in arr]
	arr2 = [x[0] for x in fc.scaler_Y.inverse_transform(fc.Ytrain)]

	if step == 0:
		out_pred = arr[-7:]
	elif step > 0:
		out_pred = ([0] * step) + arr[step-7:]
	elif step >= -7:
		out_pred = ([0] * abs(step)) + arr[-7:][:(7 + step)]
	else:
		out_pred = []

	if step == 0:
		out_current = arr2[-7:]
	elif step < 0:
		out_current = arr2[-7:]
	elif step > 7:
		out_current = []
	elif step > 0:
		out_current = arr2[step - 7:] + (step * [0])
	else:
		out_current = []

	now = datetime.datetime.now()
	# time = 

	step_a = step - 7;
	step_b = step + 7;

	date_now = now

	date_out_pred = []
	date_out_current = []

	for i in range(step + 1, step_b + 1):
		date_x = date_now + datetime.timedelta(days=i)
		date_out_pred.append(date_x.strftime("%Y-%m-%d"))

	for i in range(step_a, step):
		date_x = date_now + datetime.timedelta(days=i)
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
