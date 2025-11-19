import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime as date


df = pd.read_csv('./casoEstudio2/datos_recolectados/financialData.csv', index_col="Date_row") #reading the dataset
print(df.columns)


#dropping the duplicates if there any from the dataset
#df.drop_duplicates(inplace =True)
#print(df.shape)

#df = pd.get_dummies(df, columns = ['Open', 'High', 'Low', 'Close'], drop_first = True)

#let assingn /divide the datasetg columns in target and train that is X AND Y
# X_lapse= df.drop(columns =['Close'], axis = 1)
X =df.drop(columns =['Close', 'Adj Close'], axis = 1)
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size =0.25)
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




# Realizar las predicciones 
last_row = X.tail(1)
print(df)
X_pred = last_row #[['Open', 'High', 'Low']]
print(X_train)
print(last_row)
today = date.date.today() + pd.Timedelta(days=1) 
date_pred = today.strftime("%d/%m/%Y")   # Predecir el precio de los días siguientes
print("DATEEEEEEE ", date_pred)
y_pred = linear_model.predict(X_pred)

print("y_pred -----------")

print(y_pred)
# Crear las figuras o plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Precio de cierre real')
plt.scatter(date_pred, y_pred, color='red', marker='*', s=100, label='Predicción')

# Customizar el plot
plt.title('Precio de cierre histórico de MXN/USD y predicción')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre')
plt.legend()

# Mostrar el precio predictivo 
print('Precio predicho para', date_pred, ':', y_pred[0])

# Mostrar el plot
plt.grid(True)
plt.show()