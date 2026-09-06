import pandas as pd
import numpy as np

def r2_score(target, predict):
    m = len(target)
    ss_res = 0
    ss_tot = 0
    for i in range(m):
        ss_res += (target[i] - predict[i]) ** 2
        ss_tot += (target[i] - np.mean(target)) ** 2
    score = 1 - ss_res/ss_tot
    return score

def cost_func(x, y, w, b):
    f = x @ w + b
    total = np.mean((f - y) ** 2)
    return total

def grad_func(x, y, w, b):
    m = len(y)
    f = x @ w + b
    dc_dw = (f - y) @ x
    dc_db = np.mean(f - y)
    
    return dc_dw, dc_db

def grad_desc_f(x, y, alph=0.0001, a_in_coef=2):
    w = np.zeros(x.shape[1])
    b = 0
    cost = cost_func(x, y, w, b)
    cost_c = cost.copy()
    while cost <= cost_c:
        w_c = np.copy(w)
        b_c = np.copy(b)
        cost_c = cost.copy()
        dc_dw, dc_db = grad_func(x, y, w, b)
        w = w - alph * dc_dw
        b = b - alph * dc_db
        alph *= a_in_coef
        cost = cost_func(x, y, w, b)
        
    return w_c, b_c

class LinearRegression:
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0
    
    def fit(self, X, y, alph=0.0001, a_in_coef=2):
        self.coef_, self.intercept_ = grad_desc_f(X, y, alph, a_in_coef)
    
    def predict(self, X):
        target = X @ self.coef_ + self.intercept_
        return target