# Testing photometric redshift predictions from Machine Learning Algorithms for DESI Bright Galaxy Sample

_Keywords_ - Random forest Regression, K-nearest neighbor, Gradient Boosting, XGBoost, CatBoost, Neural Networks, Multi-Layer Perceptron, Keras, Artificial Neural Networks, Gaussian Process Regression

## Objective 

Next generation cosmology experiments like will rely on photometric redshifts rather than spectroscopic redshifts as obtaining spectroscopic data for billions of objects is both time and resource expensive. Hence, high accuracy and robust photometric redshift measurement is critical. <br>
In this study we test the performance of machine learning algorithms in predicting photometric redshifts of DESI Bright Galaxy Sample. <br>

## Data 

In this project we picked a subset of 80,952 objects with total 22 features - 20 numerical (g,r,z,w1,w2, magnitudes,g,r,z fibre magnitudes, delta chi square, extinction, sersic index, colors and shape) and 2 categorical (1.light profile morphology - radial, sersic, exponential, devoculers or point spread function and 2. sky survey region - north or south). <br>
We did some quality cuts and removed objects with any missing feature (NaN and infinity values), spectral type 'STAR' and redshifts below 0 (mostly stars) and above 0.8 (mostly Quasars) from the analysis. After these selection cuts we had total 76609 objects. 

## Methods

### 1. Random Forest Regression
This supervised learning algorithm uses ensemble learning method for regression. Ensemble learning combines predictions from multiple models to make a more accurate prediction than a single model. We use scikit-learn RandomForestRegressor package in ensemble module and test two weighting options - uniform and distance.

### 2. K-nearest Neighbor Regression
This non-parametric method approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. We use scikit-learn KNeighborsRegressor package in neighbors module.

### 3. Gradient Boosting
This ensemble machine learning algorithm is used for both classification and regression predictive modeling problems. Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This type of ensemble model referred to as boosting. Models are fit using differentiable loss function and gradient descent optimization algorithm - the loss gradient is minimized as the model is fit. We tested two types of gradient boosting algorithms - <br>

(a) XGBoost Extreme Gradient Boosting is known for its computational efficiency and often better model performance. <br>
(b) CatBoost The primary benefit of this is the support for categorical input variables in addition to computational speed improvements.<br>

### 4. Neural Networks
This algorithm mimicks signal communication in biological neurons. It is comprised of layers of nodes - an input layer, hidden layers, and an output layer. Each node connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network. We deployed two Neural Networks in the analysis - <br>

(a) Multi-Layer Perceptron (MLP) Regression : a class of feedforward artificial neural networks (ANN). Its multiple layers and non-linear activation distinguish it from a linear perceptron. It can distinguish data that is not linearly separable i.e. it can learn a non-linear function approximator. It utilizes supervised learning technique called backpropagation for training. We used a single hidden layer sequential model. <br>
(b) Keras based ANN : Keras has a high level API compared to Pytorch that is built over Tensorflow and hence making it easier to use and implement. We used a 2 hidden layers sequential model with Rectified Linear Unit (Relu) activation function and l-2 kernel regularizer.

### 5. Gaussian Process Regression
This non-parametric, Bayesian approach to regression has the ability to provide uncertainty measurements on the predictions. Unlike many popular supervised machine learning algorithms that learn exact values for every parameter in a function, the Bayesian approach infers a probability distribution over all possible values. We tested this method with Gaussian Likelihood, interpolated kernel (KISS) for calculating marginal likelihood and posterior mean and LOVE algorithm for posterior sampling and covariance matrix calculations.

### Metrics 
We used 4 metrics for the analysis and quantifying the performance and accuracy of above models - <br>
1. Normalized median absolute deviation (NMAD)
2. Root Mean Square Errors (RMSE)
3. Bias %
4. Outliers %
