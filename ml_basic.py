from sklearn import datasets,linear_model

from sklearn.metrics import mean_squared_error as msq

import numpy as np

import matplotlib.pyplot as plt

dib = datasets.load_diabetes()

dib_X = dib.data[:,np.newaxis,2]

#x_axis

dib_X_train = dib_X[:-30]

dib_X_test = dib_X[-30:]

#y_axis

dib_Y_train = dib.target[:-30]

dib_Y_test = dib.target[-30:]

#making model

model = linear_model.LinearRegression()

model.fit(dib_X_train,dib_Y_train)

dib_y_pred = model.predict(dib_X_test)

print("mean squared error: ",msq(dib_Y_test,dib_y_pred))

print("weights", model.coef_)

print("intercept: ",model.intercept_)

plt.scatter(dib_X_test,dib_Y_test)

plt.plot(dib_X_test,dib_y_pred)

plt.show()

'''

mean squared error:  3035.0601152912686

weights [941.43097333]

intercept:  153.39713623331698

#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

#print(dib.keys())

#print(dib.data)

#features_names

['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

#DESCR

.. _diabetes_dataset:

Diabetes dataset

----------------

Ten baseline variables, age, sex, body mass index, average blood

pressure, and six blood serum measurements were obtained for each of n =

442 diabetes patients, as well as the response of interest, a

quantitative measure of disease progression one year after baseline.

**Data Set Characteristics:**

  :Number of Instances: 442

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline

  :Attribute Information:

      - age     age in years

      - sex

      - bmi     body mass index

      - bp      average blood pressure

      - s1      tc, total serum cholesterol

      - s2      ldl, low-density lipoproteins

      - s3      hdl, high-density lipoproteins

      - s4      tch, total cholesterol / HDL

      - s5      ltg, possibly log of serum triglycerides level

      - s6      glu, blood sugar level

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e. the sum of squares of each column totals 1).

Source URL:

https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

For more information see:

Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.

(https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)

'''


