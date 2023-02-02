# A quick and dirty subroutine for sketching the decision area.
# Written for DS-210 at Boston University in Spring 2022.
# Author: Krzysztof Onak

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _is_integer_dtype(dtype):
    for size in ['8','16','32','64']:
        if dtype == 'int' + size:
            return True
        if dtype == 'uint' + size:
            return True
    return False

def _is_float_dtype(dtype):
    for size in ['16','32','64','128']:
        if dtype == 'float' + size:
            return True
    return False

def draw_decision_area(predictor, X, xaxis, yaxis, samples=1,
        steps=100, margin=0.05):
    # Parameters:
    #  * predictor: the classifier or regressor you trained
    #  * X: Pandas Data Frame that was used for training predictor
    #       (Don't include the target atttribute and anything that 
    #        was not used for training. All attributed have to be
    #        numerical.)
    #  * xaxis, yaxis: names of attributes to be used as coordinates
    #                  in the visualization
    #  * samples: the number of samples averaged to decide on the
    #             value of a single point in the visualization
    #  * steps: a steps * steps grid is used for visualization
    #  * margin: how much extra padding is added in the visualization
    #            in addition to the span of points in X

    # We first collect the names of columns and create a subroutine
    # for outputting a random value of that column, from the range
    # that is available in X, the input data.
    columns = []
    randomizers = []
    for col in X.columns:
        columns.append(col)
        low = X[col].min()
        high = X[col].max()
        if _is_integer_dtype(X.dtypes[col]):
            randomizers.append(lambda a=low, b=high : random.randint(a,b))
        elif _is_float_dtype(X.dtypes[col]):
            randomizers.append(lambda a=low, b=high : random.uniform(a,b))
        else:
            raise Exception("only floats and integers allowed")

    # we now compute the range values that we use for visualization
    x_min, x_max = X[xaxis].min(), X[xaxis].max()
    y_min, y_max = X[yaxis].min(), X[yaxis].max()
    delta_x, delta_y = (x_max-x_min)*margin, (y_max-y_min)*margin
    x_min -= delta_x
    y_min -= delta_y
    x_max += delta_x
    y_max += delta_y

    # Compute the grid of points for which predictions are computed
    x_coordinates = np.linspace(x_min,x_max,steps)
    y_coordinates = np.linspace(y_min,y_max,steps)
    xx, yy = np.meshgrid(x_coordinates, y_coordinates)

    # create a data frame of all samples
    total_samples = len(xx.ravel()) * samples
    df = pd.DataFrame([[f() for f in randomizers] for _ in range(total_samples)])
    df.columns = columns
    df[xaxis] = xx.repeat(samples)
    df[yaxis] = yy.repeat(samples)

    # compute the predictions and if there are multiple samples
    # per grid point, compute their mean
    colors = predictor.predict(df)
    colors = np.reshape(colors,(-1,samples))
    colors = np.array([r.mean() for r in colors])

    # plot the results
    fix,ax = plt.subplots()
    ax.contourf(xx,yy,colors.reshape(xx.shape))
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
