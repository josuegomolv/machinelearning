from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import csv
from sklearn.metrics import confusion_matrix

X = None
y = None


with open('../casoEstudio1/dataset/2d.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    X = np.array(data, dtype=np.float64) #[:,:2]

with open('../casoEstudio1/dataset/etiquetas.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    y = np.array(data, dtype=np.int64)[0]
"""¨
########################################################
            PREPROCESAMIENTO DE LOS DATOS
########################################################
"""
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


"""¨
########################################################
        Distribución del set de entrenamiento
########################################################
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test

print("y_test       ")
print(y_test)

"""¨
########################################################
        Creación de una instancia de SVM y 
        se ajustan los datos 
########################################################
"""
# controla el equilibrio entre las clasificaciones erróneas y el margen,
# es importante para crear algoritmos precisos y robustos, y evitar el sobreajuste y el subajuste.
# Valores de C más pequeños
# Un valor de C más pequeño da como resultado un margen más amplio, pero permite más clasificaciones erróneas.
# Valores de C más grandes
# Un valor de C más grande prioriza la clasificación correcta, pero da como resultado un margen más estrecho.
# Establecer el valor de C
# El valor de C generalmente se establece en uno al principio. Si se observan clasificaciones erróneas después del entrenamiento,
# el valor de C se puede ajustar utilizando el método de validación cruzada (CV) de K-fold (Divide en n partes el set de entrenamiento para n iteracioens ).

# Subajuste
# En la región de C pequeña, los modelos aprenden todos los coeficientes cero, lo que puede provocar un subajuste grave.

# Regularización
# La regularización agrega un término de penalización a la función de pérdida estándar que un modelo de aprendizaje automático 
# minimiza durante el entrenamiento. Esta penalización alienta al modelo a mantener sus parámetros pequeños, lo que puede ayudar 
# a prevenir el sobreajuste.

C = 10  # PArametro de regularización

# Creación del modelo
#El parámetro gamma define hasta dónde llega la influencia de un único ejemplo de entrenamiento:
#  Gamma bajo: significa "lejos" y da como resultado un límite más ondulado
#  Gamma alto: significa "cerca" y da como resultado un límite más suave
# Gamma es un parámetro para el núcleo RBF en SVM. 
# El parámetro gamma interviene cuando se establecen los polinomios, RBF y Sigmoid. 
# El rendimiento de una SVM se basa en sus parámetros y la función kernel utilizada.
models = (
    svm.SVC(kernel="linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=1, C=C),
    svm.SVC(kernel="poly", degree=4, gamma="auto", C=C),
)

#Asignación de datos de entrada al modelo 
models = (clf.fit(X_train, y_train) for clf in models)


"""¨
########################################################
        Diagramación de los datos  
########################################################
"""
# Asignación de titulos 
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Configuración del plot de 2x2
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_test[:, 0], X_test[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel= "Parametro1",
        ylabel= "Parametro2",
    )
    ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    #Ejecutando el test del modelo
    y_pred = clf.predict(X_test)

    # Evaluando el resultado
    resultA = metrics.accuracy_score(y_test, y_pred)
    print(title +" Accuracy:", resultA)
    print(title +" Precision:",metrics.precision_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1,2]).ravel()
    # print(tn, fp, fn, tp)  # 1 1 1 1
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



plt.show()
