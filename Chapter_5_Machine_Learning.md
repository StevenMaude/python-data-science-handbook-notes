# Notes on machine learning

## What is machine learning?

Machine learning is often categorised as a subfield of artificial
intelligence, but can be more helpful to think of machine learning as a
means of *building models of data*.

Involves building mathematical models to help understand data.
"Learning" aspect is when we give these models tunable parameters that
can adapt to observed data: the program "learns" from the data. Once
these models have been fit to previously seen data, can be used to
predict and understand aspects of newly observed data.

### Categories of machine learning

Two main types: supervised and unsupervised.

*Supervised learning* involves modelling the relationship between
measured features of data and some label associated with the data. Once
this model is determined, can be used to apply labels to new, unknown
data.  Further divide supervised learning into *classification*, where
labels are discrete categories, and *regression*, where labels are
continuous quantities.

*Unsupervised learning* involves modelling the features of a dataset
without reference to any label; "letting the dataset speak for itself".
Models include tasks such as *clustering*, identifying distinct groups
of data and *dimensionality reduction*, searching for more succinct
representations of the data.

Also, *semi-supervised learning* methods, which fall between supervised
and unsupervised learning. Can be useful if only incomplete labels are
available.

### Examples of machine learning applications

#### Classification

E.g. you are given a set of labelled points in 2D and want to use these
to classify some unlabelled points; each of these points has one of two
labels.

Have 2D data: two features for each point, represented by (x, y)
positions of points on a plane. Also have one of two *class labels* for
each point. From those features and labels, want to create a model that
decides how a newly seen point is labelled.

Simple model might be draw a line that separates the two groups of
points. The *model* is a quantitative version of the statement "a
straight line separates the classes", while the *model parameters* are
the particular numbers describing the location and orientation of that
line for our data. The optimal values for the model parameters are
learned from the data, which is often called *training the model*.

When a model is trained, can be generalised to new, unlabelled data. Can
take a new set of data, draw the same model line through it and assign
labels to the points based on this model. This stage is usually called
*prediction*.

This is the basic idea of classification, where "classification"
indicates the data has discrete class labels. May look trivial in a 2D
case, but the machine learning approach can generalise to much larger
datasets in many more dimensions.

This is similar to the task of automated spam detection for email. Might
use "spam" or "not spam" as labels, and normalised counts of important
words or phrases ("Viagra") as features. For the training set, these
labels might be determined by individual inspection of a small sample of
emails; for the remaining emails, the label would be determined using
the model. For a suitably trained classification algorithm with enough
well-constructed features (typically thousands of millions of words or
phrases), this can be an effective approach.

#### Regression: predicting continuous labels

Might instead have two features and continuous labels instead of a
discrete label.

Might use a number of regression models, but can use linear regression,
using the label as a third dimension and fitting a plane to the data.

This is similar to computing the distance to galaxies observed through a
telescope. Might use brightness of each galaxy at one of several
wavelengths as features, and distance or redshift of the galaxy as a
label. Distances for a small number of galaxies might be determined
through an independent set of observations, then a regression model used
to estimate this for other galaxies.

#### Clustering: inferring labels on unlabelled data

Clustering is a common unsupervised learning task. Data is automatically
assigned to some number of discrete groups. Clustering models use the
intrinsic structure of the data to determine which points are related.

#### Dimensionality reduction: inferring structure of unlabelled data

Seeks to pull out some low-dimensionality representation of data that in
some way preserves relevant qualities of the full dataset. For instance,
might have data with two features arranged in a spiral in a 2D plane:
could say that the data is intrinsically only 1D, although this 1D data
is embedded in higher-dimensional space. A suitable dimensionality
reduction model would be sensitive to this nonlinear structure and pull
out the lower dimensionality representation.

Important for high dimensional cases; can't visualise large number of
dimensions, so one way to make high dimensional data more manageable is
to use dimensionality reduction.

## Introducing scikit-learn

Several Python libraries provide implementations of machine learning
algorithms. scikit-learn is one of the best known, providing efficient
versions of a number of common algorithms. It is characterised by a
clean, uniform and streamlined API, as well as by useful online
documentation. A benefit of this uniformity is that once you understand
the basic use and syntax of scikit-learn for one type of model,
switching to a new model or algorithm is very straightforward.

### Data representation in scikit-learn

Machine learning is about creating models from data; for that reason,
start by discussing how data can be represented in order to be
understood by the computer. Within scikit-learn, the best way to think
about data is in terms of tables.

#### Data as table

A basic table is a 2D grid of data, where the rows represent individual
elements of the dataset, and the columns represent attributes related to
these elements.

For example, the Iris dataset available as a `DataFrame` via Seaborn:

```python
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
```

```
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
```

Each row of the data refers to a single observed flower, and the number
of rows is the total number of flowers in the dataset. In general, we
will refer to the rows of the matrix as *samples*, and the number of
rows as `n_samples`.

Likewise, each column refers to a particular quantitative piece of
information that describes each sample. In general, we will refer to the
columns of the matrix as *features* and the number of columns as
`n_features`.

##### Features matrix

The table layout makes clear that the information can be thought of as a
2D numerical array or matrix, which we will call the features matrix. By
convention, the features matrix is often stored in a variable named `X`.
The features matrix is assumed to be 2D, with shape `[n_samples,
n_features]` and is most often contained in a NumPy array or a pandas
`DataFrame`, though some scikit-learn models also accept SciPy sparse
matrices.

The samples (rows) refer to the individual objects described by the
dataset. The features (columns) refer to the distinct observations that
describe each sample in a quantitative manner. Features are generally
real-valued, but may be Boolean or discrete-valued in some cases.

##### Target array

In addition to the feature matrix `X`, we work with a *label* or
*target* array, which by convention we call `y`. The target array is
usually 1D, with length `n_samples` and generally contained in a NumPy
array or pandas `Series`. The target array may have continuous numerical
values or discrete classes/labels. While some scikit-learn estimators
handles multiple target values in the form of a 2D `[n_samples,
n_targets]` target array, here, we will be working primarily with the
common case of a 1D target array.

It can be confused how the target array differs from the other features
columns. The distinguishing feature of the target array is that it is
usually the quantity we want to *predict* from the data; in statistical
terms, it is the dependent variable. E.g. for the iris data, we may wish
to construct a model to predict the flower species from the other
measurements; here, the `species` column would be considered the target
array.

### scikit-learn's Estimator API

scikit-learn's API is designed with the following principles in mind:

* *Consistency*: All objects share a common interface drawn from a
  limited set of methods, with consistent documentation.
* *Inspection*: All specified parameter values are exposed as public
  attributes.
* *Limited object hierarchy*: Only algorithms are represented by Python
  classes; datasets are represented in standard formats (NumPy arrays,
  pandas `DataFrames`, SciPy sparse matrices) and parameter names use
  standard Python strings.
* *Composition*: Many machine learning tasks can be expressed as
  sequence of more fundamental algorithms, and scikit-learn makes use of
  this where possible.
* *Sensible defaults*: When models require user-specified parameters,
  the library defines an appropriate default value.

In practice, these principles make scikit-learn easy to use once the
basic principles are understood. Every machine learning algorithm in
scikit-learn is implemented using the Estimator API, providing a
consistent interface for a wide range of machine learning applications.

#### Basics of the API

The steps for using the Estimator API are as follows:

1. Choose a class of model by importing the appropriate estimator class
   from Scikit-Learn.
2. Choose model hyperparameters by instantiating this class with desired
   values.
3. Arrange data into a features matrix and target vector following the
   discussion above.
4. Fit the model to your data by calling the ``fit()`` method of the
   model instance.
5. Apply the Model to new data:
   - For supervised learning, often we predict labels for unknown data
     using the ``predict()`` method.
   - For unsupervised learning, we often transform or infer properties
     of the data using the ``transform()`` or ``predict()`` method.

#### Supervised learning example: simple linear regression

Consider a simple linear regression: fitting a line to (x, y) data.

Use the following simple data:

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);
```

##### Choose a class of model

Every model in scikit-learn is represented by a class. So, import the
appropriate class:

```python
from sklearn.linear_model import LinearRegression
```

##### Choose model hyperparameters

A *class* of model is not the same as an *instance* of a model.

There are still options open to us once we choose a class. We might have
to answer questions like:

* Would we like to fit for the offset (i.e., *y*-intercept)?
* Would we like the model to be normalized?
* Would we like to preprocess our features to add model flexibility?
* What degree of regularization would we like to use in our model?
* How many model components would we like to use?

These choices are often represented as *hyperparameters*; parameters
that must be set before the model is fit to data. In scikit-learn,
hyperparameters are chosen by passing values at model instantiation.

For the linear regression example, can instantiate the
`LinearRegression` class and specify that we would like to fit the
intercept using the `fit_intercept` hyperparameter:

```python
model = LinearRegression(fit_intercept=True)
```

When the model is instantiated, the only action is storing these
hyperparameter values. The model is not yet applied to any data;
scikit-learn's API makes a clear distinction between choosing a model,
and applying that model to data.

##### Arrange data into a features matrix and target vector

Our target variable `y` is in the correct form, an array of length
`n_samples`, but we need to adjust `x` to make it a matrix of size
`[n_samples, n_features]`:

```python
X = x[:, np.newaxis]
```

##### Fit the model to data

Use the `fit()` method of the model:

```python
model.fit(X, y)
```

Calling `fit()` causes a number of model-dependent internal computations
to take place, and the results of these computations are stored in
model-specific attributes that the user can explore. In scikit-learn,
by convention, model parameters learned during `fit()` have trailing
underscores, e.g. for this model we have `model.coef_` and
`model.intercept_` representing the slope and intercept of the simple
linear fit to the data. Here, they are close to the input slope of 2 and
intercept of -1.

NB: scikit-learn does not provide tools to draw conclusions from
internal model parameters themselves: interpreting model parameters is
much more a *statistical modelling* question than a *machine learning*
question. The Statsmodels package is an alternative if you wish to delve
into the meaning of fit parameters.

##### Predict labels for unknown data

Once a model is trained, the main task of supervised machine learning is
to evaluate it based on what it says about new data that was not part of
the training set. In scikit-learn, use the `predict()` method for this.
Here, our "new data" will be a grid of *x* values and we will ask what
*y* values the model predicts:

```python
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis] # coerce x values into a features matrix
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit)
```

#### Supervised learning example: Iris classification

Given a model trained on Iris data, how well can we predict the
remaining labels. Here, use a simple generative model, Gaussian naive
Bayes, that assumes each class is drawn from an axis-aligned Gaussian
distribution. Because it is fast and has no hyperparameters to choose,
Gaussian naive Bayes is a useful baseline before exploring whether
improvements can be found through more complicated models.

Would like to evaluate the model on data it has not seen before, so we
split the data into a *training set* and a *testing set*. This could be
done by hand, but scikit-learn has the `train_test_split()` function:

```python
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)
```

Now we proceed with the general pattern detailed above:

```python
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
```

Finally, we can use `accuracy_score()` to see the fraction of predicted
labels that match their true value:

```python
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
```

This model actually scores more than 97%; even a naive classification
algorithm works well for this dataset.

For classification, a confusion matrix is also a useful analysis tool to
see which classes are usually correctly or incorrectly classified;
scikit-learn provides this as `sklearn.metrics.confusion_matrix()` and
is used like `accuracy_score`, i.e. you provide `ytest` and `y_model`.

This can be plotted nicely with Seaborn's heatmap, i.e.:

```python
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');
```

#### Unsupervised learning example: Iris dimensionality

As an example, let's reduce the dimensionality of the Iris data to more
easily visualise it. The Iris data is four dimensional: four features
recorded for each sample.

Dimensionality reduction asks whether there is a suitable lower
dimensional representation that retains the essential features of the
data. This makes it easier to plot the data: two dimensions are easier
to plot, than four or more!

Here, use principal component analysis (PCA) which is a fast linear
dimensionality reduction technique. We will ask the model to return two
components: a two dimensional representation of the data.

```python
from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions
```

Can plot quickly by adding the two components to the Iris `DataFrame`
and plot with Seaborn's `lmplot`:

```python
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
```

The 2D representation has the species well separated, even though the
PCA algorithm had no knowledge of the species labels. This indicates a
straightforward classification will likely be effective on the dataset,
as we saw above with Gaussian naive Bayes.

#### Unsupervised learning example: Iris clustering

Clustering algorithms attempt to find distinct groups of data without
reference to any labels. Here, use a powerful clustering method called a
Gaussian mixture model (GMM); a GMM attempts to model the data as a
collection of Gaussian blobs.

```python
from sklearn.mixture import GMM      # 1. Choose the model class
model = GMM(n_components=3,
            covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels
```

Add the cluster label to the Iris `DataFrame` and use Seaborn to plot
the results:

```python
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False);
```

By splitting the data by cluster number, we can see how well the GMM
algorithm recovers the underlying label. Most of the species are well
separated, and there is only a small amount of mixing between versicolor
and virginica species. In other words, the measurements of these flowers
are distinct enough that we could automatically identify these different
species with a simple clustering algorithm.

## Hyperparameters

Parameters not learned from the data, parts of the model that can be
varied, but not from the data, part of initial configuration; choices
made before the model is fit to data.

## Validation

Don't want to test on data trained with: model is likely to perform much
better on this (in some cases, may be perfect, if the model just picks
the closest point).

Use holdout set: split into training set and holdout set for testing.

This is better, but still means a lot of data isn't used for training.
Cross-validation can solve this problem by splitting the data: using
part of it as a validation set.

n-fold means splitting into n groups, using one of them for validation
and the rest for training, e.g. two-fold is split in half and use half
for training, half for validation; five-fold is split into five and use
one of those groups for validation. You don't just train once, but
repeatedly, switching which group is used for validation and which
group(s) are used for training.

leave-one-out is extreme cross-validation: split into as many groups as
there are samples/observations, and train on all but one, then validate
on the one left out.

(Validation sets seem to be those used such as in cross-validation, i.e.
it's data that's not used for training directly at that point, but is
part of that corpus used in developing a model; *test* sets seem to be
those used for testing following training and validation, as a benchmark
for completely unseen data to evaluate the model.

Training sets are used to train the model; validation sets to evaluate
the model while deciding on hyperparameters (and, I assume, features
/training data set size too); test sets to test the model on unseen
data.

### Hard negative mining

Training sets can be iteratively developed; an example of this is hard
negative mining.

For instance, if you build a face recognition classifier. Testing a
classifier out on new set of images, finding the false positives and
then including those in the training set can help to improve the
classifier.

## Model selection

If our model is underperforming, what can we do?

* Use a more complicated/more flexible model
* Use a less complicated/less flexible model
* Get more training samples
* Get more data and add features to each sample

Can be counter-intuitive: more flexible model may be worse; more
training samples may not help.

## Bias-variance trade off

High bias: underfits to data (bias is referring to bias of an estimator:
difference between expected value and true value), e.g. straight line
model trained to data that fits on a curve; e.g. model too simplistic
for the data.

High variance: overfits the data; model accounts for random errors in
the data as well as the underlying data distribution, reflects the noise
in the data, not the real process creating that data; model fits the
data too closely.

High bias gives similar performance on validation and training sets.

High variance model typically performs worse on validation sets than on
training set.

### Validation curve

Plotting scores against complexity (e.g. polynomial degree in polynomial
regression) is a validation curve.

Increase complexity: training score increases towards some a limit,
validation score increases too (as bias decreases) then drops off as
variance increases. High bias at low complexity. Best model is somewhere
in the middle of complexity.

Training score higher than validation everywhere: model typically fits
data better if already seen the data.

Low model complexity (high bias): training data is underfit, so model is
a poor predictor for training and unseen data.

High model complexity (high variance): training data is overfit, so
model predicts training data well, but fails for unseen data.

Validation curve at a maximum at some intermediate value: a trade-off
between bias and variance.

### Learning curve

Optimal model may also depend on size of training data as well as
complexity, so validation curve behaviour depends on these too.

A learning curve is a plot of training/validation score against size of
the training set.

Larger data can support a more complex model, more resilient to
overfitting.

Expect that learning curves behave as follows:

* A model of a given complexity will overfit a small dataset; training
  score is relatively high, while validation score is relatively low.
* A model of a given complexity will underfit a large dataset; training
  score will decrease, but validation score will increase.
* A model will never, except by chance, give a better score to the
  validation set than the training set; so the two curves should get
  closer together, but not cross over.

Also note that learning curve tends to converge to a particular score as
training set size increases: adding more training data doesn't help. In
that case, to increase model performance, need a different (often more
complex) model.

High complexity models may give a better convergent score than lower
complexity models, but require a larger dataset to prevent overfitting
and get close to that convergent score.

### Grid search

Simple example was polynomial degree, but models often have a number of
knobs to turn, so instead of simple 2D plots, can get multidimensional
surfaces that describe their behaviour.

Finding best model can be done via grid search: explore a grid of model
features, varying them and calculating score, e.g. in scikit-learn,
GridSearchCV. Can parallelise search, search at random to help find
suitable hyperparameters.

If you find the best values are at the edges of the grid, might want to
consider extending the grid to ensure those best values are really
optimal.

## Feature engineering

Not all data is numerical. Feature engineering is taking information
that you have and turning it into numbers to build a feature matrix.
Also called vectorisation: turning data into vectors.

### Categorical data

Might be tempted to encode a category as a number, but often for models
implies a numerical ordering, e.g. assigning numbers to places or
colours would imply that "red > blue", for instance or "Manchester <
London".

The solution is one-hot encoding: so create columns that represent
presence or absence of a category (so they can contribute individually
to some estimator).

This is explained quite well on [Stack
Overflow](https://stackoverflow.com/questions/17469835/why-does-one-hot-encoding-improve-machine-learning-performance).

(The difference from conventional numerical features e.g. if you were
looking at e.g. width of petal, then you'd have a weight for that part
of the model, plus a contribution from the magnitude of the observed
value, which gives you some result.)

Can result in many columns, but can use sparse matrices to store more
efficiently.

### Text features

Can encode text as word counts. Create a feature matrix where each
sample (row) is an individual text, the columns represent words, and the
values represent word counts.

Can also use term frequency-inverse document frequency (tf-idf) which
weights the word count features by how often they appear in documents.

### Image features

One option is using a pixel value, but may not be optimal.

### Derived features

Create features from input features.

One useful example is constructing polynomial features from input data
to then perform linear regression.

Makes linear regression into polynomial regression by transforming the
input: basis function regression.

For example, take x, y data and create a feature matrix of x^3, x^2, x.
The explanation of the maths behind this is not clear from this book
section, but the clever part here is that linear regression can work for
multiple dimensional inputs, and we choose those dimensions to be x,
x^2, x^3.

### Imputation of missing data

Fill missing data with some value to apply model to data. Different
options, e.g. use mean of column or some other model.

scikit-learn has `Imputer` class for simple approaches, e.g. mean,
median, most frequent value. Can then feed this into estimator.

### scikit-learn pipelines

Can create pipelines to save manually carrying out multiple steps, e.g.
impute missing values as the mean, transform to quadratic polynomial
features, then fit a linear regression.

```
from sklearn.pipeline import make_pipeline

model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
```

Can then use that just another scikit-learn model, via `model.fit()`,
`model.predict()` etc.

## Naive Bayes

Fast and simple classification algorithms often suitable for very
high-dimensional datasets. Few tunable parameters, so quick baseline for
classification problems.

### Bayesian Classification

Naive Bayes classifiers are built on Bayesian classification methods, in
turn, relying on Bayes' theorem.

In Bayesian classification, want to find probability of a label given
some observed features: P(L|features).

Bayes' theorem allows us to calculate this in terms of quantities we can
calculate more directly:

```
P(L|features) = P(features|L) P(L)
                ------------------
                    P(features)
```

If trying to decide between two labels — L1 and L2 — then one way to
decide is to calculate the ratio of the posterior probabilities for each
label:

```
P(L1|features) = P(features|L1) P(L1)
--------------   --------------------
P(L2|features) = P(features|L2) P(L2)
```

Need a model to find P(features|Li) for each label, Li. A model is
generative because it specifies the hypothetical random process that
generates the data. Specifying this model for each label is the main
piece of the training of such a classifier. The general version of this
step is difficult, but can be simplified via assumptions about the form
of the model.

This is the "naive" part: making naive assumptions about the generative
model for each label, can find a rough approximation of the model and
then proceed with classification.

Different naive Bayes classifiers rest on different naive assumptions
about the data.

### Gaussian naive Bayes

For this classifier, assume that data from each label is drawn from a
simple Gaussian distribution.

Find the mean and standard deviation of the points in each label. Builds
a Gaussian generative model with larger probabilities near the centre of
those distributions. Can then calculate the likelihood P(features|Li)
and the posterior ratio to find the label.

Often get a quadratic boundary in Gaussian naive Bayes.

Can also use Bayesian classification to estimate probabilities for the
classes.

### Multinomial naive Bayes

Can use multinomial naive Bayes: assume the features are from a
multinomial distribution, one which describes the probability of
observing counts among different categories.

Particularly appropriate for features representing counts or count
rates, e.g. word counts or frequencies.

Similar to Gaussian naive Bayes, just with a different generative model.

### When to use naive Bayes

Naive Bayesian classifiers make stringent assumptions about data:
generally don't perform as well as a more complex model.

Advantages:

* Fast for training and prediction.
* Straightforward probabilistic prediction.
* Often easy to interpret.
* Few (if any) tunable parameters.

Good initial choice. If it works well, can use as a fast, easy to
understand classifier for the problem. If not, can use more complex
models with some baseline of how well they should perform.

Work well in the following situations:

* When naive assumptions match the data (rarely in practice).
* For very well-separated categories, when model complexity is less
  important.
* For very high-dimensional data, when model complexity is less
  important.

The last two items are related: as the dimensions grow, much less likely
for two points to be close together (as they must be close in every
dimension). Clusters in high dimensions tend to be more separated than
those in low dimensions, on average, if the new dimensions add
information.

So, simplistic classifiers, like naive Bayes, tend to work as well or
better than more complex ones as dimensionality grows: simple models can
be powerful, given enough data.

## Linear regression

Naive Bayes is a good starting point for classification, and linear
regression is a good starting point for regression. These can be fit
quickly and are very interpretable.

The familiar model is a straight line data fit, but can be extended to
model more complex data.

Can handle multidimensional linear models, e.g.  fitting a plane to
points in three dimensions or hyperplanes to points in higher
dimensions.

### Basis function regression

As mentioned above, can use linear regression for nonlinear
relationships between variables by transforming the data according to
basis functions, replacing the x1, x2, x3... in a linear model with e.g.
x, x^2, x^3.

The linearity refers to the fact that the coefficients don't multiply or
divide each other. This projects one-dimensional x values into a higher
dimension, to fit more complex relationships between x and y.

As above, can use polynomial features, but can use other basis
functions, which can be created by a user even if not available in
scikit-learn, e.g. Gaussian basis functions.

### Regularisation

Basis functions can make linear regression models more flexible, but
lead to overfitting, especially if using too many basis functions. What
can happen is that coefficients of basis functions (e.g. adjacent
Gaussian basis functions) can blow up, cancelling each other out.

Regularisation can limit these spikes by penalising large values of the
model parameters. The penalty parameter should be determined via
cross-validation.

#### Ridge regression (L2 regularisation)

Penalises the sum of squares of the model coefficients with a parameter
that multiplies this sum and controls the strength of the penalty.

As this penalty approaches the limit of zero, we get back the standard linear
regression model; as it approaches the limit of infinity, the model
responses will be suppressed.

Ridge regression can be computed efficiently, with little cost over the
standard linear regression model.

#### Lasso regression (L1 regularisation)

Penalises the sum of absolute values. Similar to ridge regression in
concept, but can given very different results: tends to favour sparse
models, that set model coefficients to zero.

## Support vector machines (SVMs)

SVMs are a powerful and flexible class of supervised algorithms for
classification and regression. Here, we'll discuss their use for
classification.

### Motivation

For Bayesian classification above, we learned a generative model
describing the distribution of each class and used that to predict
labels for new points. That is generative classification.

Here, we consider discriminative classification: instead of modelling
each class, we find a line or curve (in 2D) or manifold (in multiple
dimensions) that divides the classes.

For separating two groups of points belonging to different classes, it
may be that a linear discriminative classifier creates a straight line
that divides the two sets of data, and creates a model for
classification. However, there may be more than one line that satisfies
this criterion. Furthermore, different lines may give different
predictions for the same point.

### SVMs maximise the margin

Instead of drawing a zero-width line between classes, draw a margin
around each line of some width, up to the nearest point.

For SVMs, choose the line that maximises this margin as the optimal
model: a maximum margin estimator.

Training points that touch the margins are the pivotal elements of the
fit: these are the support vectors. Only the position of the support
vectors matters: points further from the margin on the correct side do
not affect the fit. Those points do not contribute to the loss function
used to fit the model. So, even adding points, provided they're further
away than the margin, won't affect the fit.

### Kernel SVM

Have seen kernels before in basis function regressions above. Projecting
data into higher-dimensional space defined by different basis functions
and then fitting nonlinear relationships with a linear classifier.

Can do same for SVM. For example, imagine data of two classes with one
class in the centre of two features, and the other class data
surrounding it in a rough circle. A linear separator won't work.

However, if you add a dimension using a radial basis function, centre
positioned class (with smaller radius) will have different values in
this dimension to the outer class (with larger radius). This makes them
easy to separate by a plane, parallel to x and y, in the middle of the
radial basis function values.

Depending on how the function was chosen, may not be so cleanly
separable. Choosing the function is a problem: automating this is ideal.
One way is to compute a basis function at every point in the dataset and
let the algorithm choose the best result. This type of basis function
transformation is a kernel transformation, based on a similarity
relationship (or kernel) between each pair of points.

However, projecting N points into N dimensions could be computationally
expensive. Instead, use the kernel trick, which allows a fit on
kernel-transformed data to be done implicitly, without creating the full
N dimensional representation.

(For scikit-learn, can apply kernelised SVM by changing the linear
kernel to a radial basis function kernel, via `kernel` hyperparameter.)

Kernel transformation is often used in machine learning to turn fast
linear methods into fast nonlinear methods, especially for models where
the kernel trick can be used.

### Tuning SVM: softening margins

If no perfect decision boundary exists and data overlaps, can use a
tuning parameter, C that allows points to enter the margin, if that
allows a better fit. For very large C, the margin is hard and points
cannot lie in it. For smaller C, the margin is softer and can grow to
include some points.

### Pros and cons of SVM

SVM is powerful for classification because:

* Depend on relatively few support vectors so are compact models and
  take up little memory.
* When model is trained, prediction is fast.
* Only affected by points near margin, so work well with
  high-dimensional data, even data with more dimensions than samples,
  which can be challenging for other models.
* Integration with kernel methods makes them adaptable to many types of
  data.

But:

* Scaling with number of sample is O(N^3) at worst, or O(N^2) for
  efficient implementations. For large training sets, can be expensive
  computationally.
* Results depend strongly on a suitable choice for softening parameter
  C. Need to find this via cross-validation, which can be expensive for
  large datasets.
* Results do not have a direct probabilistic interpretation. Can be
  estimated via an internal cross-validation, but can be costly.

Useful if faster, simpler, less tuning-intensive methods are
insufficient. If processing time available, can be an excellent choice.

## Decision trees and random forests

Random forests are a nonparametric algorithm. They are an example of an
ensemble method: relying on aggregating the results of an ensemble of
simpler estimators. With ensemble methods, the sum can be greater than
its parts: a majority vote among estimators can be better than any of
the individual estimators.

### Decision trees

Random forests are ensemble learners built on decision trees.

Decision trees classify objects by asking a series of questions to
zero-in on their classification, each question with (typically, if not
always) two mutually exclusive answers.

Binary splitting makes this efficient: each question will cut the number
of options by approximately half. The trick is deciding the questions to
ask at each step.

In machine learning implementations of decision trees, the questions
generally take the form of axis-aligned splits in the data: each node in
the tree splits the data into two groups using a cutoff value within one
of the features.

In a simple case of two features, a decision tree iteratively splits the
data along one or other axis according to some quantitative criterion.
At each level, assign the label of the new region according to the
majority vote of the points within it.

At each level of the tree, each region is split along one or the other
features, unless it wholly contains points of one class (there's no need
to continue splitting them as the voting result is the same).

`DecisionTreeClassifier` in scikit-learn.

#### Decision trees and overfitting

As depth increases, can get strange shaped regions, due to overfitting.

This is a common property of decision trees: end up fitting the details
of the data, not the overall property of the distribution the data are
drawn from.

Get inconsistencies between trees looking at different data. Turns out
that using information from multiple trees is a way to improve our
result.

### Random forests

This leads to the idea of bagging: using an ensemble of parallel
estimators that each overfit the data, and averages the result. An
ensemble of randomised decision trees is a random forest.

`RandomForestClassifier` in scikit-learn. Only need to select the number
of estimators, but can work very well.

#### Random forest regression

Can use `RandomForestRegressor` in scikit-learn for regression.

#### Pros and cons of random forests

Advantages:
* Training and prediction are fast, because of the simplicity of the
  decision trees. Both training and prediction can be parallelised
  easily as the trees are entirely independent.
* Multiple trees allow for probabilistic classification: majority vote
  among estimators allows probability to be estimated.
* Nonparametric model is flexible and can perform well on tasks that are
  underfit by other estimators.

Disadvantage:
* Not easy to interpret the results; meaning of the classification model
  not easy to draw conclusions from.

## Principal component analysis (PCA)

A commonly used unsupervised algorithm.

Used for dimensionality reduction, but also for visualisation, noise
filtering, feature extraction and engineering.

It is a fast and flexible unsupervised method for dimensionality
reduction in data. Unsupervised learning aims to learn about the
relationship, e.g. between x and y values, not predict y from x.

For PCA, this relationship is quantified by finding the principal axes
in the data, and using those to describe the dataset. Can visualise as
vectors on the input data: using components to define the vector
direction and the explained variance to define the squared length of the
vector. The length of the vector indicates the "importance" of that axis
in describing the data distribution: it is a measure of the variance of
the data when projected onto that axis.

Transformation from data axes to principal axes is an affine
transformation: consisting of a translation, rotation and uniform
scaling.

PCA has lots of uses.

### PCA as dimensionality reduction

In dimensionality reduction with PCA, zero out one or more of the
smallest principal components, resulting in a lower dimensional
projection of the data that preserves the maximal data variance.

Can reduce dimensionality, then inverse transform the reduced data to
see the effect of this reduction. Loses information along least
important axis or axes, leaving the components with the highest
variance. The fraction of variance lost is roughly a measure of the
information lost in this process.

A reduced dimension dataset can be still useful enough to describe the
important relationships between data points.

### PCA for visualisation

In high dimensions, can use a low dimensional representation to
visualise the data in e.g. two dimensions. As if you find the optimal
stretch and rotation in that high dimensional space to see the
projection in two dimensions, and can be done without reference to the
labels.

Consider digits as 8x8 pixel representations. Could represent in terms
of pixel values, with a pixel basis where each pixel is an individual
dimension; could multiply each pixel value by the pixel it describes to
reconstruct the image.

(I think this is a bit confusing, because a zero value would typically
be a black pixel, not a white one, but it gets the idea of
dimensionality reduction across.)

Could also reduce the dimensionality by using the first eight pixels
only, but would lose a lot of the image and these wouldn't represent it
very well.

However, can choose a different basis, e.g. functions that include
pre-defined contributions from each pixel, and add just a few of those
together to better reconstruct the data.

PCA finds these more efficient basis functions.

### Choosing number of components

Look at cumulative explained variance ratio as a function of the number
of components: see how much of the information is kept as number of
components increases. May be able to keep most of the variance with a
relatively small number of components.

### PCA as noise filtering

Components with variance much larger than effect of noise should be
relatively unaffected by noise, so reconstructing data using the largest
subset of principal components, you should preferentially keep the
signal and discard the noise. Can specify how much of variance to retain
in PCA with scikit-learn.

Perform PCA fit on data, transform the data and then use the inverse of
the transform to reconstruct the filtered digits.

PCA can be useful for feature selection since you can train a classifier
on lower dimensional reduction of high dimensional data, which will help
to filter out random noise.

### Summary of PCA

Useful in a wide range of contexts. Useful to visualise relationship
between points, to understand variance in data and to understand the
dimensionality (looking at how many components you need to represent the
data).

Weakness is that it can be strongly affected by data outliers. Robust
variants act to iteratively discard data points that are poorly
described by the initial components, e.g. `RandomizedPCA` and
`SparsePCA` in scikit-learn.

## Manifold learning

PCA can be used to reduce dimensionality: reducing the number of
features of a dataset while maintaining the essential relationships
between the points. It is fast and flexible, but does not perform well
where there are nonlinear relationships within the data.

Can use manifold learning methods instead. These are unsupervised
estimators that aim to describe datasets as low dimensional manifolds
embedded in high dimensional spaces.

To think of a manifold, can imagine a sheet of paper: a two dimensional
object that exists in a three dimensional world, and can be bent or
rolled in those two dimensions.

Rotating, reorienting or stretching the paper in three dimensional space
doesn't change the geometry of the paper: these are akin to linear
embeddings. If you bend, curl or crumple the paper, it is a two
dimensional manifold still, but the embedding in three dimensional space
is no longer linear.

Manifold learning algorithms aim to learn about the fundamental two
dimensional nature of the paper, even as it is contorted to fill the
three dimensional space.

### Multidimensional scaling (MDS)

For some data, e.g. points in the shape of the word "HELLO" in 2D, the
x and y values are not the most fundamental description of the data.

Can rotate or scale the data and the "HELLO" will still be apparent.

What is fundamental is the distance between each point and the other
points. Can represent using a distance matrix: construct an NxN array
such that each entry (i, j) contains the distance between point i and
point j, e.g. scikit-learn's `pairwise_distances` function.

Even if this "HELLO" data is rotated or translated, get the same
distance matrix. However, the distance matrix is not easily visualised.
Furthermore, it is difficult to convert the distance matrix back into
x and y coordinates.

MDS can be used to reconstruct a D-dimensional coordinate representation
of the data (i.e. the number of coordinates in the distance matrix),
only given the distance matrix.

#### MDS as manifold learning

Distance matrices can be computed from data in any dimension, and can
reduce down to fewer components (e.g. 3D to 2D).

Given high dimensional embedded data, it seeks a low dimensional
representation of the data that preserves certain relationships within
the data. For MDS, the preserved quantity is the distance between pairs
of points.

### Nonlinear embeddings

MDS breaks down when embeddings are nonlinear, e.g. "HELLO" distorted
into an S-shape in three dimensions, by curving the text around.

MDS cannot unwrap this nonlinear embedding, so lose the relationships
when it is applied.

#### Locally linear embedding (LLE)

MDS attempts to preserve distances between faraway points. Instead, LLE
aims to preserve distance between neighbouring points. Can be better at
recovering well-defined manifolds with little distortion.

### More on manifold methods

Challenges of manifold learning:

* No good framework for handling missing data, but straightforward
  iterative approaches for this case in PCA.
* Presence of noise in the data can drastically change the embedding.
  PCA filters noise from the most important components.
* Manifold embedding result depends on the number of neighbours chosen,
  and no good quantitative way to choose this. PCA doesn't need this
  choice.
* The globally optimal number of output dimensions is difficult to
  determine. PCA lets you choose this based on explained variance.
* Meaning of embedded dimensions is not always clear. In PCA, the
  principal components have a clear meaning.
* Manifold learning methods scale as O(N^2) or O(N^3). For PCA, there
  are randomised approaches that are much faster.

The advantage of manifold learning methods over PCA is that they can
preserve nonlinear relationships in the data, but PCA is often a better
first choice.

For toy problems, LLE and variants perform well. For high dimensional
data from real world sources, IsoMap tends to give more meaningful
embeddings than LLE. For highly clustered data, t-distributed stochastic
neighbor embedding can work well, but can be slow.

## k-means clustering

Clustering algorithms seek to learn from the data an optimal division or
discrete labelling of the data.

Many clustering algorithms, but k-means is simple to understand.

### Introducing k-means

Looks for a pre-determined number of clusters within an unlabelled
multidimensional dataset. Optimal clustering as considered here is
simple:

* The "cluster centre" is the arithmetic mean of all the points
  belonging to the cluster.
* Each point is closer to its own centre than to others.

These assumptions are the basis of the k-means model.

Finds clusters quickly, even though the number of possible combinations
of cluster assignments could be large. However, doesn't use an
exhaustive search, but expectation-maximisation, an iterative approach.

### k-means algorithm: expectation-maximisation (E-M)

E-M is a powerful algorithm. k-means is a simple application of it; the
E-M approach here consists of:

1. Guess some cluster centres.
2. Repeat until converged:
       A. E-step: assign points to the nearest cluster centre
       B. M-step: set the cluster centres to the mean

The E-step is called that because we update our expectation of which
cluster each point belongs to. The M-step is named such because it
involves maximising some fitness function that defines the location of
the cluster centres, here by taking the mean of the data in each
cluster.

Usually, repeating the E-step and M-step will result in a better
estimate of the cluster characteristics.

#### Caveats of E-M and k-means

* Although repeated E-M improves the clusters, the globally optimal
  result may not be achieved. Usually run it for multiple starting
  guesses.
* Need to tell it how many clusters are expected: this can't be learned
  from the data. Whether clustering result is meaningful is difficult to
  answer: can use silhouette analysis. Otherwise, can use other
  clustering algorithms that do have a measure of fitness per number of
  clusters, e.g. Gaussian mixture models, or which can choose a suitable
  number of clusters (DBSCAN, mean-shift or affinity propagation).
* k-means is limited to linear cluster boundaries; assumes points are
  closer to their cluster centre than others. Can fail if clusters have
  complicated geometries. Can work around this by using kernelised
  k-means, e.g. `SpectralClustering` in scikit-learn: transforms data
  into higher dimensions and then assigns labels using k-means, e.g.
  like linear regression above where data transformed so that more
  complex curves can be used in fitting.
* Slow for large numbers of samples. Every iteration of k-means accesses
  every data point. Can solve by using a subset of the data to update
  the cluster centres, via batch-based k-means algorithms, e.g.
  `MiniBatchKMeans` in scikit-learn.

### Other uses of k-means

Aside from clustering for classification, also have novel uses, e.g.
colour compression by compressing the colour values in pixels to reduce
the data; replace a group of pixels with a single cluster centre.

## Gaussian mixture models (GMMs)

k-means is simple and easy to understand, but its simplicity leads to
practical challenges for its application. In particular, the
non-probabilistic nature of k-means and its use of simple
distance from cluster centre to assign cluster membership leads to poor
performance for many real world situations.

### Weaknesses of k-means

#### No probability measure

k-means has no intrinsic measure of probability or uncertainty of
cluster assignments (though may be possible to use a bootstrap approach
to estimate the uncertainty), e.g. in regions where clusters overlap.

#### Inflexible cluster shape

Can consider k-means as placing a circle (or, in higher dimensions, a
hypersphere) at the centre of each cluster, with a radius defined by the
most distant point in the cluster. This radius acts as a hard cutoff for
cluster assignment within the training set: any point outside the circle
is not a member of the cluster.

This circular structure means that k-means has no way of accounting or
oblong or elliptical clusters; circular cluster may be a poor fit, but
k-means will still try to fit the data to circular clusters. Can get a
mixing of cluster assignments where these circles overlap.

### Generalising E-M: Gaussian mixture models

Might imagine addressing these two weaknesses by generalising the
k-means model, e.g. measure uncertainty in cluster assignment by
comparing the distances of each point to all cluster centres, not just
the closest. Might also consider making the cluster boundaries
elliptical to account for non-circular clusters. These are the essential
components of an alternative clustering model: Gaussian mixture models.

A GMM attempts to find a mixture of multidimensional Gaussian
probability distributions that best model any input dataset.

In the simplest case, GMMs can be used to find clusters, just as k-means
does. But, as it contains a probabilistic model, it can find
probabilistic cluster assignments (`predict_proba` method in
scikit-learn), which give the probability of a point belonging to the
given cluster.

Like k-means, uses expectation-maximisation approach which does the
following:

1. Choose starting guesses for the location and shape.
2. Repeat until converged:
        A. E-step: for each point, find weights encoding the probability
           of membership in each cluster.
        B. M-step: for each cluster, update its location, normalisation
           and shape based on all data points, making use of the
           weights.

The result is that each cluster is associated with a smooth Gaussian
model, not a hard-edged sphere. Just as in the k-means E-M approach,
GMMs can sometimes miss the globally optimal solution, and multiple
random initialisations are typically used.

#### Choosing the covariance type

Covariance type is a hyperparameter that controls the degrees of freedom
in the shape of each cluster. This is an important setting. The
scikit-learn default is `covariance_type='diag'`, which means that the
size of the cluster along each dimension can be set independently, with
the resulting ellipse constrained to align with the axes.

A slightly simpler and faster model is `spherical`; this model
constrains the shape of the cluster such that all dimensions are equal.
The resulting clustering will have similar characteristics to k-means,
although it is not entirely equivalent.

A more complicated and more expensive model (especially as the number of
dimensions increases) is `full` which allows each cluster to be modelled
as an ellipse with arbitrary orientation.

### GMM as density estimation

GMM is often thought of as clustering algorithm, but is fundamentally an
algorithm for density estimation. The result of a GMM fit to data is
technically not a clustering model, but a generative probabilistic model
describing the data distribution.

For data not in "nice" clusters, e.g. a distribution shaped in
two interleaved crescents, using two component GMM gives a poor fit:
where the distribution is clustered into two halves, ignoring the
crescents.

Using many more components and ignoring cluster labels, however, get a
fit closer to the input data. Models the distribution of input data: a
generative model that we can use to generate new random data distributed
similarly to the input.

#### How many components?

GMM being a generative model gives a natural way of determining the
optimal number of components for a given dataset. A generative model is
inherently a probability distribution for the dataset, so we can
evaluate the *likelihood* of the data under the model, using
cross-validation to avoid overfitting (likelihood being roughly: a
measure of the extent that the data supports the model).

Can also correct for overfitting by adjusting the model likelihoods
using some criterion like the Akaike information criterion (AIC) or the
Bayesian information criterion (BIC). scikit-learn's GMM model includes
methods to computer these. Optimal number of clusters minimises the AIC
or BIC. This choice of number of components measures GMM's performance
as a density estimator, not as a clustering algorithm. Better to think
of GMM as a density estimator, and use it for clustering only when
warranted with simple datasets.

### GMM for generating new data

Can use GMM, e.g. with digits data, to synthesise new examples of
digits. Note that for high dimensional spaces, GMMs can have trouble
converging, so using PCA and preserving most of the variance (e.g. 99%
in the cited example) is a useful first processing step to reduce the
number of dimensions with minimal information loss.

In scikit-learn, use `sample` method for new examples, and then do the
inverse transform of the PCA to get back data in the form that's useful
(i.e. to display them).

## Kernel density estimation (KDE)

A density estimator is an algorithm which takes a D-dimensional dataset
and produces an estimate of the D-dimensional probability distribution
that data is taken from.

The GMM algorithm does this by representing the density as a weighted
sum of Gaussian distributions. KDE goes further and uses a single
Gaussian component per point, making it essentially a non-parametric
estimator of density.

### Histograms

A density estimator models the probability distribution that generated a
dataset. For 1D data, a histogram is a familiar simple density
estimator. A histogram divides data into discrete bins, counts the
points that fall in each bin and visualises the results.

Can create a normalised histogram where the height of the bins reflects
probability density. However, an issue with using a histogram as density
estimator is that the choice of bin size and location can lead to
representations with different features: e.g. small shift in bin
locations can nudge values into neighbouring bins and modify the
distribution.

Can also think of histogram as a stack of blocks, where one block is
stacked within each bin on top of each point in the dataset. Instead of
stacking the blocks aligned with the bins, can stack the blocks aligned
with the points they represent. The blocks from different points won't
be aligned, but can add their contributions at each location along the
x-axis to find the result. This gives a more robust reflection of the
data characteristics than the standard histogram, although gives rough
edges that don't reflect true properties of the data. Smoothing these
out by using a function like a Gaussian at each point gives a much more
accurate idea of the shape of the distribution with much less variance
(changes much less in response to sampling differences).

### Kernel density estimation in practice

The free KDE parameters are the kernel, specifying the distribution
shape placed at each point, and the kernel bandwidth, which controls the
size of the kernel at each point. There are many kernels you might use
for a KDE.

#### Selecting bandwidth via cross-validation

Bandwidth of KDE is important to finding a suitable density estimate,
and controls the bias-variance trade-off: too narrow a bandwidth leads
to a high variance estimate (overfitting), where the presence or absence
of a single point makes a large difference; too wide a bandwidth leads
to a high bias estimate (underfitting) where the structure in the data
is washed out by the wide kernel.

Can find this via cross-validation.

### KDE uses

#### Visualisation

Instead of plotting individual points representing individual
observations on a map, may plot kernel density instead to give a clearer
picture.

#### Bayesian generative classification

Can do Bayesian classification but removing the "naive" part, part by
using a more sophisticated generative model for each class.

The approach is generally:

1. Split the training data by label.
2. For each set, fit a KDE to give a model of the data. This allows you
   for an observation x and label y to compute a likelihood P(x|y).
3. From the examples of each class in the training set, compute the
   class prior P(y).
4. For an unknown point x, the posterior probability for each class is
   P(y|x) ∝ P(x|y)P(y); the class maximising this posterior is the label
   assigned to the point.
