This is multiple linear regression, made from scratch using python.

For this project I used the gradient descend method.

There we have two files:
  'model.py'
  'predicting.py'

In 'model.py' I have 4 functions:
  cost_func - counts loss
  grad_func - some scalar maths
  grad_desc_f - counts gradient descent for model
  r2_score - counts accuracy of a model

and class LinearRegression with 2 methods: .fit and .predict

In 'predicting.py' you just use this model similar as in sklearn.

Fun fact: this model has higher accuracy than sklearn model
