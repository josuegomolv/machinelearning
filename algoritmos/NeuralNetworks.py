from sklearn.neural_network import MLPClassifier
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

X = None
y = None


with open('../casoEstudio1/dataset/datosSinEtiquetas.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    X = np.array(data, dtype=np.float64) #[:,:2]

with open('../casoEstudio1/dataset/etiquetas.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    y = np.array(data, dtype=np.int64)[0]

    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test
print(y_test)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 1), random_state=1, max_iter = 3)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

# Evaluando el resultado
resultA = metrics.accuracy_score(y_test, y_pred)
print(" Accuracy:", resultA)
print(" Precision:",metrics.precision_score(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1,2]).ravel()

print("     -------------------------------------")
    
print("          Positivo            Negativo     ")
print("     -------------------------------------")
print("Posi |                |                   |")
print("tivo |      ",tp ,"      |        ",fp ,"      |")
print("     |                |                   |")
print("     -------------------------------------")
print("Nega |                |                   |")
print("tivo |      ",fn ,"      |        ", tn,"      |")
print("     |                |                   |")
print("     -------------------------------------")
#pred_proba = clf.predict_proba([[2., 2.], [-1., -2.]])
#print(pred_proba)