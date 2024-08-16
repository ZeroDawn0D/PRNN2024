# PRNN2024


The three assignments required for E1 213 Pattern Recognition and Neural Networks (Jan 2024) by Prathosh AP

TODO: Rewrite all outside of classroom requirement

Each assignment+viva contributes 15% of the final grade

Doing everything mentioned will give you 70% of the 15 points. Extra experiments for 100%

*Note: For A2, LibSVM is not enough, implement SVM from scratch with cvxopt library*

*Note: Use cross-validation for hyperparameter tuning* 

## Assignment 1

### Regression
- Multi-linear Regression (3 targets and 10 features)
- Generalised Regression with polynomial kernel (3 targets and 2 features)
- Generalised Regression with non-polynomial kernel (1 target(probability) and 5 features)

### Classification
- Binary Classification (10 features and 2 classes)
- Multiclass Classification (25 features and 10 classes)

#### Implementations
- **Bayes' classifiers** with 0-1 loss assuming Normal, Exponential and GMMs (with diagonal co-variances) as class-conditional densities. For GMMs, code up the **EM algorithm**
- Bayes' with non-parametric density estimator (**Parzen Window**) with 2 different kernels
- **K-Nearest Neighbour** with different K-values and 2 different distance metrics (euclidean and cosine distances)
- **Linear classifier** (One vs Rest incase of multiclass)

#### Metrics
- Classification accuracy
- Confustion Matrix
- Class-wise F1 score
- RoC curves for any pair of classes
- likelihood curve for EM with different choices for the number of mixtures as hyper parameters
- Empirical risk on the train and test data while using logistic regressor

## Assignment 2

### Neural Networks
- Implement error backpropagation algorithm for Fully Connected feed-forward neural network and multilayer convolutional neural network. Hyperparameters: loss_function, input_dimension, num_hidden_layers, num_layer_nodes, num_kernels, kernel_size, padding,stride
- Set hyperparameters to overfit data
- Use atleast 3 regularisation techniques and plot the bias-variance curve

### Support Vector Machines
- Implement SVMs both with and without slack formulations. Experiment with 3 kernels and grid-search on hyperparameters. Use LibSVM and compare result with CVXOPT. Use OvR for multiclass.

#### Implementations
- Implement regression from A1 using Multi-layer perceptrons (MLPs)
- Implement classification from A1 using MLP and repeat the experiment with at least two loss functions
- Implement multi-class classification of Kuzushiji-MNIST with Logistic Regression, SVM with Gaussian Kernel, MLP and CNN. Compare performances

## Assignment 3

### Tasks
- Implement a **self-attention block** from scratch and use it with an MLP. Hyperparameters: token_length, num_attention_layers
- Implement **PCA** with hyperparameter: projected_dimensionality
- Implement **K-Means Clustering** with different distance metrics as hyper-parameters
- Implement a **Decision Tree Classifier** and a **Random Forest** with hyperparameters: multipler impurity functions
- Implement **Gradient Boosting** that can take multiple classifiers as inputs and performs assembling

### Implementation
#### Vision
- Solve 10 class classification with CNN
- MLP on PCA'd features
- Transformer model on PCA'd features
- K-means on both raw and PCA'd features
- Ensemble of CNN/MLP/Decision Trees in an **Adaboost framework** and compare results
- Metrics: Accuracy, F1 Sccore and NMI for clustering.
#### Text
Use **TF-IDF embeddings**. 12 class classification problem. Solve using MLP, Transformer with self-attention and Random Forest and Gradient Boosted Tree.


