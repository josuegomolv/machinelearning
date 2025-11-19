import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime as date


df = pd.read_csv('./casoEstudio3/datos_recolectados/insurance.csv') #reading the dataset
print(df.columns)


df.drop_duplicates(inplace =True)
df.shape



#let encode some columns from text to numeric format using the ONE HOT ENCODER
df = pd.get_dummies(df, columns = ['sex', 'smoker', 'region'], drop_first = True)
print(df)
#let assingn /divide the datasetg columns in target and train that is X AND Y
X =df.drop(columns =['charges'], axis = 1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size =0.30)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_model.score(X_test, y_test)

linear_predictions = linear_model.predict(X_test)
print(linear_predictions)

linear_rmse = np.sqrt(mean_absolute_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("Linear Regression:")
print("RMSE:", linear_rmse)
print("R-squared:", linear_r2)

#linear regression 
plt.scatter(y_test, linear_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--')
plt.title('Linear Regression - Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.show()

# Mostrar el plot
plt.grid(True)
plt.show()