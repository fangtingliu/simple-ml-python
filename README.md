# MachineLearning_FLP

MachineLearning_FLP is a Python module that includes common MachineLearning algorithms. They are implemented according to books and courses I have taken. Some of them include test with data provided by those books and courses. Sources are specified in the detailed described below. Thanks to all those sources.
Courses:
- 'Machine Learning by Standford University' courses by Andrew Ng, hosted by Coursera
- Artificial Intelligence for Humans Volume 1: Fundamental Algorithms by Jeff Heaton

## Algorithms
### Linear Regression Model APIs
* loadtxt(path): load txt documents as input data, it treats the last value of each line as y
* plot(X, Y, markerType, title, xlabel, ylabel): only applicable with one feature, i.e. X is a vector
* gradientDescent(X, Y, learning_rate, interations_number): returns fitted thetas and historial cost values of iterations
* linearR(path, learning_rate, interations_number, learning_curve=True): returns fitted thetas and historial cost values of iterations. It also plots fitting line if only one feature provided. Learning curve plotting is optional, see example for detials
* normalizeFeature(X): when features in model have variance scale, normalizing features return normalized values of data, mean vector for each feature and standard deviation for each feature
* predict(thetas, X, mean, std): return predicted value for input X based on linear model. Mean and std are defaulted to 0 and 1

The following APIs are only for one feature model:
* plotFitLine(X, Y, thetas): plot fitting line when only one feature in the model
* costSurfPlot(X, Y, theta0_start, theta0_end, theta1_start, theta1_end): surf plot on cost function on different thetas. Recommend selecting start and end values based on fitted thetas
* costContourPlot(X, Y, theta0_start, theta0_end, theta1_start, theta1_end):

## Usages
```bash
pip install -r requirements.txt
```

# Linear Regression Model

### One feature example (y = theta0 + theta1 * x)
```python
from linearR import linearR

linearR.linearR("linearData.txt", 0.01, 1500, learning_curve=True)
```
or
```python
X, Y = linearR.loadtxt("linearData.txt")

thetas, JHist = linearR.gradientDescent(X, Y, 0.01, 1500)
linearR.plot(range(1500), JHist, "b-", "Cost on Number of Iterations")

linearR.plotFitLine(X, Y, thetas)
linearR.costSurfPlot(X, Y, -10, 10, -1, 4)
linearR.costContourPlot(X, Y, thetas, -10, 10, -1, 4)
```

### Multiple features example (y = theta0 + theta1 * x_1 + theta_2 * x_2 + ...)
```python
from linearR import linearR

X, Y = linearR.loadtxt("linearMultiData.txt")
X, mean, std = linearR.normalizeFeature(X)
thetas, JHist = linearR.gradientDescent(X, Y, 0.88, 50)
print("Thetas found by gradient descent:", thetas)
linearR.plot(range(50), JHist, "b-")
print(linearR.predict(thetas, ([1650, 3] - mean) / std))
```

* Normal equation example
```python
from linearR import linearR

X, Y = linearR.loadtxt("linearMultiData.txt")
thetas = linearR.normalEq(X, Y)
print("Thetas found by gradient descent:", thetas)
print(linearR.predict(thetas, [1650, 3]))
```

# K-Means classification Model (non-supervised)
* K is number of clusters
* Default iteration is 50

### Example
```python
import numpy as np

from kMeans import kMeans

data = np.loadtxt({path}, delimiter=",", dtype="float")
kMeans.kMeans(data, K, iter_num)

kMeans.kMeansImage({path}, K, iter_num)
#image must locate at the same directory of the file running kmeans
```

