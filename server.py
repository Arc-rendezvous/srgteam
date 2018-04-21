import __future__
from Forecaster import Forecaster
import os
from flask import Flask, request, json, render_template, Response
from flask_cors import CORS
import json

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
	return json.dumps(arr)

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5003)
