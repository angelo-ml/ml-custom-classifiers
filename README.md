# Machine Learning Custom Classifiers
This library contains custom implementations of some popular machine learning classifiers. Currently it includes the logistic regression, decision tree and random forrest classifiers. More will be added soon.

## Example Usage
Assuming that we have loaded the data X_train, y_train, X_test, y_test and X_predict.

For the Logistic Regression:
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

For the Decision Tree:
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

For the Random Forrest:
```
# Import the library
from custom_classifiers import RandomForrest

# Build Classifier Object
clf = RandomForrest(n_estimators=100, max_features=None, max_depth=None, min_samples_split=2)

# train the model
clf.fit(X_train, y_train)

# Test model's accuracy
accuracy = clf.score(X_test, y_test)

# predict values for new data
pred = clf.predict(X_predict)
```

## Input parameters

- **alpha:** Learning rate of the logistic regression.
- **max_depth:** Maximum depth of the decision trees (default value: None).
- **min_samples_split:** The minimum number of samples required to split an internal node (default value: 2).
- **n_estimators:** Number of decision tree estimators for the random forrest (default value: 100).
- **max_features:** The number of features to consider for each split (default value: sqrt(n_features)).
