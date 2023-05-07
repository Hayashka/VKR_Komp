import flask
from flask import render_template
import pickle
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

app =  flask.Flask(__name__, template_folder= 'templates')

@app.route('/', methods = ['POST', 'GET'])
@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        angle = int(flask.request.form['angle'])
        step = float(flask.request.form['step'])
        density = float(flask.request.form['density'])
        elasticity_module = float(flask.request.form['elasticity_module'])
        hardener_quantity = float(flask.request.form['hardener_quantity'])
        epoxy_group = float(flask.request.form['epoxy_group'])
        temperature = float(flask.request.form['temperature'])
        surface_density = float(flask.request.form['surface_density'])      
        elasticity_module2 = float(flask.request.form['elasticity_module2'])
        tensile_strength = float(flask.request.form['tensile_strength'])
        resin_consumption = float(flask.request.form['resin_consumption'])
        patch_density = float(flask.request.form['patch_density'])

        y_pred = loaded_model.predict([[angle, step, patch_density, density, elasticity_module, hardener_quantity, epoxy_group, temperature, surface_density, elasticity_module2, tensile_strength, resin_consumption]])

        return render_template('main.html', result = y_pred)

if __name__ == '__main__':
    app.run(debug=True)
 