from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels:', np.unique(y))
iris

import pandas as pd

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Verify that the data was stratified correctly with 70% in train and 30% in test
np.bincount(y)
np.bincount(y_train)
np.bincount(y_test)

# %% codecell
# We standardize the input in chapter 2 with numpy mean and std methods
# Here, we standardize features with StandardScaler class from sklearn preprocessing module
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# fit method estimates the parameters μ (sample mean) and σ (standard deviation) for each feature dimension from the training data
sc.fit(X_train)
# transform method standardizes the training and test data using the estimated parameters
X_train_std  = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Instead of making our own perceptron, we use sklearn's model intead
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# We can calculate classification accuracy of perceptron on the test set with metrics like accuracy score
from sklearn.metrics import accuracy_score
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# We can also use the score method of the classifier
print('Accuracy: {:.3f}'.format(ppn.score(X_test_std, y_test)))

# Use plot_decision_regions function to visualize how well this model separates the different flower samples. 
# A small modification is added to highlight the samples from the test dataset via small circles:
# %% codecell
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx], 
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', 
                    edgecolor='black', 
                    alpha=1.0,
                    linewidth=1, 
                    marker='o', 
                    s=100, 
                    label='test set')
X_combined_std = np.vstack((X_train_std, X_test_std))                    
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]') 
plt.ylabel('petal width [standardized]') 
plt.legend(loc='upper left')
plt.show()
