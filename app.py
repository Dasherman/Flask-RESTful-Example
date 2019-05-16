# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:24:58 2019

@author: Don Vu
"""

from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle
import numpy as np
from io import StringIO
import pandas as pd
from model_training import kNN

def read_data(string):
    #Do not include column names
    f = StringIO(string)
    data = pd.read_csv(f, header=None, names=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    X = data.drop(columns=['sex', 'smoker', 'region'])
    X['sex_male'] = data['sex']=='male'
    X['smoker_yes'] = data['smoker']=='yes'
    X['region_northwest'] = data['region']=='northwest'
    X['region_southeast'] = data['region']=='southeast'
    X['region_southwest'] = data['region']=='southhwest'
    
    
    return np.array(X)

def array2string(array):
    return np.array2string(array, separator=',').replace('[', '').replace(']', '').replace('\n', '')

#standard stuff
app = Flask('test')
api = Api(app)

#load model
with open('model.pickled', 'rb') as f:
    model = pickle.load(f)

#argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

#Resource object for prediction
class PredictCharge(Resource):
    #Do not include column names in request
    def get(self):
        #find query
        args = parser.parse_args()
        query = args['query']
        
        X = read_data(query)
        
        pred = model.pred(X)
        
        return {'predictions': array2string(pred)}
    
api.add_resource(PredictCharge, '/')   

if __name__ == '__main__':
    app.run(debug=True)