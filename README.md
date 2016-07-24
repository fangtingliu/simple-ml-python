# MachineLearning_FLP

MachineLearning_FLP is a Python module that includes common MachineLearning algorithms. Some of them are implemented according to ML books, some of them are from courses I have taken. Some of then include test with data provided by those books and courses. Sources are specified in the detailed described below. Thanks to all those sources.

## Algorithms
* Linear Regression Model
The implementation of Linear Regression Model is based on the 'Machine Learning by Standford University' courses by Andrew Ng, hosted by Coursera.

## Usages
# Linear Regression Model

* One feature example (y = theta0 + theta1 * x)
```python
X, Y = linearR.loadtxt("linearData.txt") 
linearR.plot(X, Y, "r+", "TrainingData", "Population of City in 10,000s", "Profit in $10,000s")
thetas, JHist = linearR.gradientDescent(X, Y, 0.01, 1500)
print("Thetas found by gradient descent:", thetas)
linearR.plot(range(1500), JHist, "b-", "Cost on Number of Iterations")
linearR.plotFitLine(X, Y, thetas)
linearR.costSurfPlot(X, Y, -10, 10, -1, 4)
linearR.costContourPlot(X, Y, thetas, -10, 10, -1, 4)
```

* Multiple features example (y = theta0 + theta1 * x_1 + theta_2 * x_2 + ...)
```python
X, Y = linearR.loadtxt("linearMultiData.txt")
X, mean, std = linearR.normalizeFeature(X)
thetas, JHist = linearR.gradientDescent(X, Y, 0.88, 50)
linearR.plot(range(50), JHist, "b-")
print("Thetas found by gradient descent:", thetas)
print(linearR.predict(thetas, ([1650, 3] - mean) / std))
```

* Normal equation example
```python
X, Y = linearR.loadtxt("linearMultiData.txt")
thetas = linearR.normalEq(X, Y)
print("Thetas found by gradient descent:", thetas)
print(linearR.predict(thetas, [1650, 3]))
```

