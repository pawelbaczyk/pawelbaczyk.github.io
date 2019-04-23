---
layout: post
title:  "Looking at trees in a forest"
date:   2019-04-18 21:41:09 +0200
categories: ML
author: Paweł
---

We are going to study a classification problem on a small, black and white image. The colors are going to be represented with two categories: black as 0 and white as 1. The image contains a quadrilateral which looks like this:

![png](/assets/plot1.png)

First, let's train a decision tree, `dt`, to predict if a pixel is black or white (dependent variable `y`). Independent variables, `X`, are two Cartesian coordinates of the pixel. We want to study predictions of models which are not perfectly fitted to the data, so we restrict the maximum depth of the tree. In the fitting procedure all features are taken into account as `max_features=None` (this is the default option for `DecisionTreeClassifier` given here only for completeness). The error of the model is calculated in % as `mean_absolute_error`.

```python
dt = DecisionTreeClassifier(max_depth=8, max_features=None)
dt.fit(X, y)
err = mean_absolute_error(y, dt.predict(X)) * 100 # error in %
```

The prediction of the fitted decision tree is presented in the figure below together with the calculated error:

![png](/assets/plot2.png)

As intended we obtain an inaccurate model with an error of around 3%. Let's try something more sophisticated and train a random forest classifier, `rf` with 20 decision trees as estimators. Similarly, the model is fitted using all features at each split as `max_features=None`. Interestingly, this is NOT the default option for `RandomForestClassifier` (the default one is taking the square root of `n_features`). To introduce randomness, each tree of the forest is fitted with bootstrapped sample.

```python
rf = RandomForestClassifier(n_estimators=20, max_depth=8, bootstrap=True, max_features=None)
rf.fit(X,y) 
err = mean_absolute_error(y, rf.predict(X)) * 100 # error in %
```

The prediction of such a random forest model is shown below:

![png](/assets/plot3.png)

As expected the calculated error is lower than for single decision tree. This is achieved by averaging predictions of 20 estimators which are presented below:

![png](/assets/plot4.png)

Every single tree in this set is less accurate than the first decision tree we trained (with error of about 3%). This is caused by the bootstrapping we employed while fitting. However, the trees trained in this way are not fully correlated and therefore generalize well while calculating random forest prediction (error of about 1%).

Lastly, we can play with the restriction imposed on our models that is `max_depth` parameter. On the figure below, we can see three lines representing errors calculated for different `max_depth` for single decision tree, random forest with 20 estimators and average error for these 20 estimators. We can see that random forest has better accuracy then single decision tree for shallow trees. Both errors go to zero as the depth of the tree is increased. The average error of an estimator of random forest is almost the same as for a single decision tree. Due to bootstrapping it does not go to zero as `max_depth` is increased.

![png](/assets/plot5.png)

