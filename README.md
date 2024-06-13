# Getting started

## Prerequisites
* scikit-learn

## Instalation

`pip install scikit-learn`

## Usage

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

`from sklearn.ensemble import RandomForestClassifier` import Random Forest Classifier

### Initialize, train and predict

```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
```

`clf = RandomForestClassifier(n_estimators=100, random_state=42)` Initialize clf with n_estimators=100, this is a number of trees
