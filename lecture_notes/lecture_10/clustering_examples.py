import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def generate_data():
    x1 = np.random.normal(size=100)/10
    y1 = np.random.normal(size=100)/10
    x2 = 1.0 + np.random.normal(size=100)/10
    y2 = 1.0 + np.random.normal(size=100)/10
    x3 = 1.0 + np.random.normal(size=30)/10
    y3 = np.random.normal(size=30)/10
    x4 = np.random.normal(size=30)/10
    y4 = 1.0 + np.random.normal(size=30)/10
    
    x = np.concatenate((x1,x2,x3,x4))
    y = np.concatenate((y1,y2,y3,y4))
    return (x,y)

x,y = generate_data()

def plot(c):
    plt.figure(figsize=(8,7))
    plt.scatter(x,y,10,"b");
    plt.scatter([c[0],c[2]],[c[1],c[3]],100,"r");

def get_distances(c):
    q = (x - c[0])**2 + (y - c[1])**2
    r = (x - c[2])**2 + (y - c[3])**2
    return np.sqrt(np.minimum(q,r))

def kmeans():
    sol1 = least_squares(get_distances,[0.1,0.0,0.9,1.0])
    sol2 = least_squares(get_distances,[0.0,0.1,1.0,0.9])
    if sol1.fun[0] < sol2.fun[0]:
        plot(sol1.x)
    else:
        plot(sol2.x)

def kmedian():
    sqrt_distances = lambda c : np.sqrt(get_distances(c))
    sol1 = least_squares(sqrt_distances,[0.1,0.0,0.9,1.0])
    sol2 = least_squares(sqrt_distances,[0.0,0.1,1.0,0.9])
    if sol1.fun[0] < sol2.fun[0]:
        plot(sol1.x)
    else:
        plot(sol2.x)

def kcenter():
    sqrt_distances = lambda c : get_distances(c).max()
    sol1 = least_squares(sqrt_distances,[0.5,0.0,0.5,1.0])
    sol2 = least_squares(sqrt_distances,[0.0,0.5,1.0,0.5])
    if sol1.fun[0] < sol2.fun[0]:
        plot(sol1.x)
    else:
        plot(sol2.x)
