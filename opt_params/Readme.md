
# Optimizing parameters for SVM
To optimize the usage of SVM, its parameters have to be chosen carefully in advance. We use Gaussian Kernal SVM which has two parameters to be tuned. The first parameter is the regularization parameter , which determines the trade-off cost between minimizing the training error and minimizing the complexity of the model. The second parameter is gamma  which defines the non-linear mapping from the input space to some high-dimensional feature space. 

## Grid search OF SVM hyper-parameters optimization
We have tried a wide range of C and Gamma . For each dataset we have used 15 different values of C and  Gamma. The 15 different values of resulting in a total of 225 pairs of (C, Gamma) in each fold. The 15 different values of C are 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000, 10000 and the 15 different values of  Gamma  are 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.8, 1, 2, 5, 10, 20, 100 and 1000. Using 5-fold cross-validation on the training set and LIBSVM toolbox 

![Test Image 4](https://github.com/hjleed/Project1_part/blob/master/opt_params/parameters_grid.png)

## vim: set fileencoding=<opt_hyper_params.py> :

Optimizing parameters:   closedset/opt_hyper_params.py
The function is looking for search grid optimization looping for different values of C and Gamma.
 I used:     C = [0.001,  0.01,  0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 10, 20, 50, 100, 1000, 10000]  
      &  Gamma = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 5,  10, 20, 100, 1000, 10000]
Description: looping for all combination of C and Gamma, this function computes 5-fold classification.

### Editing

* Line 52 (LIBSVM toolbox location)

* Line 59 (Features location)

