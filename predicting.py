#There you just use things to predict
import pandas as pd
import numpy as np
from model import LinearRegression
# from sklearn.datasets import load_diabetes
# from model import r2_score

# dia = load_diabetes(as_frame=True)
# df = dia.frame
# df

# train, test = np.split(df.sample(frac=1), [int(0.8 * len(df))])
# train = pd.DataFrame(train, columns = [df.columns.values.tolist()])
# test = pd.DataFrame(test, columns = [df.columns.values.tolist()])

# X_train, y_train = train[train.columns[:-1]].values, train[train.columns[-1]].values
# X_test, y_test = test[test.columns[:-1]].values, test[test.columns[-1]].values

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# print(f'accuracy is: {r2}')