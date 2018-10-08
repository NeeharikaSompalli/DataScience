import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def pca_calc(data, number_of_components, eigen_values, eigen_vectors):
    idx = np.argsort(eigen_values)[::-1][:number_of_components]
    eigen_values, eigen_vectors = eigen_vectors[idx], eigen_vectors[:, idx]
    transformed_data = np.dot(data, eigen_vectors)
    return transformed_data


standardize = lambda x: (x - x.mean()) / x.std()


data_train = pd.read_csv('hw2q1_train.csv')
data_test = pd.read_csv('hw2q1_test.csv')
data_train['Class'] = data_train['Class'].map({'R': 1, 'M': 0})
data_test['Class'] = data_test['Class'].map({'R': 1, 'M': 0})

# a) size of the training and testing sets
print ("training size: ", data_train.shape)
print ("testing size: ", data_test.shape)

# a) Rock (R) and Mine (M) samples are in the training set and the testing set
print ("# of Rock (R) samples : ", len(data_train.loc[data_train['Class'] == 1]))
print ("# of Mine (M) samples : ", len(data_train.loc[data_train['Class'] == 0]))
print ("# of Rock (R) samples : ", len(data_test.loc[data_test['Class'] == 1]))
print ("# of Mine (M) samples : ", len(data_test.loc[data_test['Class'] == 0]))
print('\n')

y_train = data_train.iloc[:, -1]
y_test = data_test.iloc[:, -1]

# normalize
train_df = pd.DataFrame(data_train.iloc[:, 0:60])
min_train = train_df.min()
max_train = train_df.max()
X_train = (train_df - min_train) / (max_train - min_train)

test_df = pd.DataFrame(data_test.iloc[:, 0:60])
X_test = (test_df - min_train) / (max_train - min_train)

# Calculate Covarince
covarinace_matrix = X_train.cov()
print(covarinace_matrix)

# Calculate Eigen values
w, v = linalg.eig(covarinace_matrix)
print("Eigen Values\n", w)
print("Eigen Vector\n", v)
print("Size of Covarinance matrix\n", covarinace_matrix.shape)
top_five = np.sort(w)[::-1][0:5]
print("Top 5 eigen values\n", top_five)

x = np.arange(1, 31)
plt.title("Eigen Plot")
plt.bar(x, w[:30])
plt.ylabel('Eigen values')
plt.xlabel('Eigen Vector Number')
plt.savefig("eigen_normal.png")
plt.clf()

transformed_train = pca_calc(X_train, 10, w, v)
transformed_test = pca_calc(X_test, 10, w, v)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(transformed_train, y_train)
y_pred = knn.predict(transformed_test)
print('Accuracy: ', accuracy_score(y_test, y_pred), '\n')
result = pd.DataFrame(transformed_test)
result['Acutal class'] = y_train
result['Predicated class'] = y_pred
result.to_csv('result_normal.csv')

components = [2, 4, 8, 10, 20, 40, 60]
accuracies = []

for component in components:
    transformed_train = pca_calc(X_train, component, w, v)
    transformed_test = pca_calc(X_test, component, w, v)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(transformed_train, y_train)
    accuracy = knn.score(transformed_test, y_test)
    accuracies.append(accuracy)

# print(accuracies)
plt.title("accuracy plot")
plt.plot(components, accuracies)
plt.ylabel('accuracy')
plt.xlabel('no of components')
plt.savefig("accuracy_normal.png")
plt.clf()

# Standardized data

X_train = standardize(data_train.iloc[:, 0:60])
X_test = standardize(data_test.iloc[:, 0:60])

# Calculate Covarince
covarinace_matrix = X_train.cov()
print(covarinace_matrix)

# Calculate Eigen values
w, v = linalg.eig(covarinace_matrix)
print("Eigen Values\n", w)
print("Eigen Vector\n", v)
print("Size of Covarinance matrix\n", covarinace_matrix.shape)
top_five = np.sort(w)[::-1][0:5]
print("Top 5 eigen values\n", top_five)

x = np.arange(1, 61)
plt.title("Eigen")
plt.bar(x, w)
plt.ylabel('eigen values')
plt.xlabel('no of components')
plt.savefig("eigen_standard.png")
plt.clf()

transformed_train = pca_calc(X_train, 10, w, v)
transformed_test = pca_calc(X_test, 10, w, v)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(transformed_train, y_train)
y_pred = knn.predict(transformed_test)
print('Accuracy: ', accuracy_score(y_test, y_pred), '\n')
result = pd.DataFrame(transformed_test)
result['Acutal class'] = y_train
result['Predicated class'] = y_pred
result.to_csv('result_standard.csv')

components = [2, 4, 8, 10, 20, 40, 60]
accuracies = []

for component in components:
    transformed_train = pca_calc(X_train, component, w, v)
    transformed_test = pca_calc(X_test, component, w, v)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(transformed_train, y_train)
    accuracy = knn.score(transformed_test, y_test)
    accuracies.append(accuracy)

# print(accuracies)
plt.title("accuracy plot")
plt.plot(components, accuracies)
plt.ylabel('accuracy')
plt.xlabel('no of components')
plt.savefig("accuracy_standard.png")
plt.clf()
