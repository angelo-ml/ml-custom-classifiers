# Machine Learning Custom Classifiers
This is a library containing custom implementations of some popular machine learning classifiers. Currently it includes the logistic regression and decision tree classifiers. More will be added soon.

## Example Usage
Assuming that we have loaded the data X_train, y_train, X_test, y_test and X_predict.

For the logistic regression:
```
# Import the library
from custom_classifiers import LogisticRegression

# Build Classifier Object
clf = LogisticRegression(alpha=0.001)

# train the model
clf.fit(X_train, y_train)

# Test model's accuracy
accuracy = clf.score(X_test, y_test)

# predict values for new data
pred = clf.predict(X_predict)
```

For the decision tree:
```
# Import the library
from custom_classifiers import DecisionTree

# Build Classifier Object
clf = DecisionTree(max_depth = None, min_samples_split=2)

# train the model
clf.fit(X_train, y_train)

# Test model's accuracy
accuracy = clf.score(X_test, y_test)

# predict values for new data
pred = clf.predict(X_predict)
```