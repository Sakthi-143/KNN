# KNN

# Glass Classification using KNN

## Introduction
This repository contains code for preparing a model to classify different types of glass using the K-Nearest Neighbors (KNN) algorithm. The dataset includes various features such as refractive index, sodium content, magnesium content,etc., which are used to predict the type of glass.

## Data Description
The dataset consists of the following features:
- RI: Refractive index
- Na: Sodium (weight percent in corresponding oxide)
- Mg: Magnesium
- Al: Aluminum
- Si: Silicon
- K: Potassium
- Ca: Calcium
- Ba: Barium
- Fe: Iron
- Type: Type of glass (class attribute)

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Data Understanding
The dataset comprises 214 entries with 10 columns. There are no missing values, and all columns are numeric.

## Data Visualization and Exploration
Visualization techniques like scatter plots, heatmaps, and pair plots are used to understand the relationships between different features and the target variable.

## Feature Scaling
Standard scaling is applied to scale down the features to unit variance, which is necessary for distance-based algorithms like KNN.

## Applying KNN
KNN algorithm is implemented using the scikit-learn library. Various parameters like the number of neighbors (k) and distance metrics (Euclidean, Manhattan) are experimented with to find the best model.

## Finding the Best K Value
Cross-validation and error rate analysis are performed to determine the optimal value of k. The accuracy of the model is evaluated, and the best k value is found to be 4.

## Results and Conclusion
- Manhattan distance metric produced better results compared to Euclidean distance.
- Feature scaling improved accuracy by almost 5%.
- Dropping 'Ca' feature slightly improved the results.
- The model achieved an accuracy of approximately 73.84% in classifying glass types.

## Animal Classification using KNN
Additionally, the repository includes code for implementing a KNN model to classify animals into categories based on their features.

## Conclusion
Both KNN models demonstrate the effectiveness of the algorithm in classification tasks, highlighting the importance of parameter tuning and feature selection for optimal results.



## Linear Regression Readme

This repository contains code implementing Linear Regression along with other related techniques such as Cross Validation, Ridge Regression, Lasso Regression, Random Forest Classifier, Logistic Regression, Support Vector Machine (SVM), K-Means Clustering, Dendogram, t-distributed Stochastic Neighbor Embedding (t-SNE), Principal Component Analysis (PCA), and visualization techniques like scatter plots, confusion matrices, and bar plots.

### Implementation Details:

#### Linear Regression:

The linear regression model is implemented using scikit-learn's `LinearRegression` class. The model is trained on the given data and then used to predict outcomes. 

##### Plotting Regression Line and Scatter:

The regression line along with the scatter plot is plotted to visualize the linear relationship between the features and the target variable.

##### Cross Validation:

Cross-validation is performed to evaluate the performance of the linear regression model. The data is split into `k` folds, and the model is trained and evaluated `k` times. The average score across all folds is reported as the cross-validation score.

#### Ridge Regression:

Ridge regression, a regularization technique, is implemented using scikit-learn's `Ridge` class. The data is split into training and testing sets, and standard scaling is applied. Ridge regression is then performed, and the score of the model is reported.

#### Lasso Regression:

Lasso regression, another regularization technique, is implemented using scikit-learn's `Lasso` class. The data is split into training and testing sets, and Lasso regression is performed. The score and coefficients of the model are reported.

#### Random Forest Classifier:

Random Forest Classifier is implemented using scikit-learn's `RandomForestClassifier` class. The data is split into training and testing sets, and the model is trained and evaluated. The confusion matrix and classification report are generated to evaluate the model's performance.

#### Logistic Regression:

Logistic Regression is implemented using scikit-learn's `LogisticRegression` class. The data is split into training and testing sets, and logistic regression is performed. The ROC curve is plotted to evaluate the model's performance.

#### Support Vector Machine (SVM):

Support Vector Machine is implemented using scikit-learn's `SVC` class. Standard scaling is applied to the data, and a pipeline is created to preprocess and fit the SVM model. The accuracy and tuned model parameters are reported.

#### K-Means Clustering:

K-Means Clustering is implemented using scikit-learn's `KMeans` class. The data is clustered into `k` clusters, and scatter plots along with cross-tabulation tables are generated to visualize and analyze the clustering.

#### Dendogram:

A dendrogram is plotted using scipy's `dendrogram` function to visualize the hierarchical clustering of the data.

#### t-distributed Stochastic Neighbor Embedding (t-SNE):

t-SNE is implemented using scikit-learn's `TSNE` class. The data is transformed into lower-dimensional space for visualization purposes.

#### Principal Component Analysis (PCA):

PCA is implemented using scikit-learn's `PCA` class. The variance explained by each principal component is visualized using a bar plot, and PCA is applied to reduce the dimensionality of the data for visualization.

### Usage:

To use this code, simply run the provided Jupyter Notebook or Python script. Ensure that all necessary libraries are installed in your environment. You can modify the code as needed for your specific dataset and analysis requirements.

### References:

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
- [Numpy Documentation](https://numpy.org/doc/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/index.html)
