from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import csv
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

X = None
y = None


with open('../casoEstudio1/dataset/K2D.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    X = np.array(data, dtype=np.float64) #[:,:2]

with open('../casoEstudio1/dataset/etiquetas.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    y = np.array(data, dtype=np.int64)[0]


min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test

kmeans = KMeans(n_clusters=2, random_state=3, n_init="auto", max_iter= 10).fit(X)

y_pred = kmeans.predict(X_test)

print("y_pred -----------------------------")
print(y_pred)

print(y_test)
y_test = y_test > 1 
print("y_pred -----------------------------")

print(y_test)

h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(X_test[:, 0], X_test[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())



# Evaluando el resultado
resultA = metrics.accuracy_score(y_test, y_pred)
print(" Accuracy:", resultA)
print(" Precision:",metrics.precision_score(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

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



plt.show()

