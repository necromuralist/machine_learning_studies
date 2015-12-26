
# coding: utf-8

# # Regression Week 4: Ridge Regression (interpretation)

# In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
# * Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression
# * Use matplotlib to visualize polynomial regressions
# * Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression, this time with L2 penalty
# * Use matplotlib to visualize polynomial regressions under L2 regularization
# * Choose best L2 penalty using cross-validation.
# * Assess the final fit using test data.
# 
# We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# ## Fire up graphlab create

# In[398]:

# python standard library
from collections import namedtuple

# third party
import graphlab
import matplotlib.pyplot as plot
import numpy
import seaborn


# In[399]:



# In[400]:

class Constants(object):
    __slots__ = ()
    target = 'price'
    predictor = 'sqft_living'
    base_polynomial = 'power_1'
# end class Constants


# ## Polynomial regression, revisited

# We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:

# In[401]:

def polynomial_sframe(feature, degree):
    """
    :param:
     - `feature`: vector of degree 1 data
     - `degree`: highest degree to raise feature data
    :return: Frame with feature raised up to degree
    """
    assert degree >= 1, "`degree` must be >= 1, not {0}".format(degree)
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe[Constants.base_polynomial] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree + 1): 
            # first we'll give the column a name:
            name = 'power_{0}'.format(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe    


# In[402]:

test_data = graphlab.SArray(range(100))
try:
    output = polynomial_sframe(test_data, -1)
    raise Exception('this should not be reachable')
except AssertionError:
    pass
degree = 3
output = polynomial_sframe(test_data, degree)
assert type(output) == graphlab.SFrame
for i in range(degree):
    column = 'power_{0}'.format(i + 1)
    assert column in output.column_names()
    assert len(output[column]) == len(test_data)
    assert output[column] == test_data**(i + 1)


# Let's use matplotlib to visualize what a polynomial regression looks like on the house data.

# In[403]:

sales = graphlab.SFrame('../../large_data/kc_house_data.gl/')


# As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.

# In[404]:

sales = sales.sort([Constants.predictor, Constants.target])


# Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:

# In[405]:

l2_small_penalty = 1e-5


# In[406]:

poly_15 = polynomial_sframe(sales[Constants.predictor], 15)
features = poly_15.column_names()
poly_15[Constants.target] = sales[Constants.target]


# *Note*: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  (In lecture, we discussed the fact that regularization can also help with numerical stability, and here we are seeing a practical example.)
# 
# With the L2 penalty specified above, fit the model and print out the learned weights.
# 
# Hint: make sure to add 'price' column to the new SFrame before calling `graphlab.linear_regression.create()`. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set=None` in this call.

# In[407]:

model_15 = graphlab.linear_regression.create(poly_15,
                                             target=Constants.target,
                                             features=features,
                                             validation_set=None,
                                             l2_penalty=l2_small_penalty,
                                             verbose=False)


# In[408]:

coefficients = model_15.get('coefficients')
print(coefficients)


# In[409]:

print(coefficients.column_names())


# In[410]:

coefficients[coefficients['name'] == 'power_1']['value'][0]


# ***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***

# **answer:** 103.090949754

# ## Observe overfitting

# Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

# First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 

# In[411]:

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)


# Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.
# 
# Hint: When calling `graphlab.linear_regression.create()`, use the same L2 penalty as before (i.e. `l2_small_penalty`).  Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.

# In[412]:

sets = [set_1, set_2, set_3, set_4]
set_data = {'set_{0}'.format(count+1): data for count, data in enumerate(sets)}
polynomials = [polynomial_sframe(data[Constants.predictor], 15) for data in sets]
for index, frame in enumerate(polynomials):
    frame[Constants.target] = sets[index][Constants.target]
polynomial_data = {'set_{0}'.format(count+1): data for count, data in enumerate(polynomials)}
    
def get_models(penalty=l2_small_penalty):
    return {name: graphlab.linear_regression.create(data,
                                                    target=Constants.target,
                                                    features=features,
                                                    validation_set=None,
                                                    l2_penalty=penalty,
                                                    verbose=False)
            for name, data in polynomial_data.items()}
models = get_models()


# In[413]:

def get_coefficients(models):
    return {name: model.get('coefficients') for name, model in models.items()}

model_coefficients = get_coefficients(models)
for name, coefficients in model_coefficients.items():
    print(name)
    print(coefficients)


# In[414]:

seaborn.set_style('whitegrid')

def plot_models(models, penalty, palette='colorblind'):
    seaborn.set_palette(palette)
    for name, model in models.items():
        figure = plot.figure()
        axe = figure.gca()
        x = polynomial_data[name][Constants.base_polynomial]
        lines = axe.plot(x, set_data[name][Constants.target], '.', label='data')
        lines = axe.plot(x, model.predict(polynomial_data[name]), '-', label='model')
        legend = axe.legend(loc="upper left")
        title = axe.set_title("Sq Ft Living vs Price ({0}, degree 15, penalty {1})".format(name, penalty))
        label = axe.set_xlabel("Living Area (Sq Ft)")
        label = axe.set_ylabel("Price ($)")
plot_models(models, l2_small_penalty)


# The four curves should differ from one another a lot, as should the coefficients you learned.
# 
# ***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# In[415]:

def smallest_largest_coefficient(model_coefficients):
    smallest = min(coefficients[coefficients['name'] == Constants.base_polynomial]['value'][0] for coefficients in model_coefficients.values())
    largest = max(coefficients[coefficients['name'] == Constants.base_polynomial]['value'][0] for coefficients in model_coefficients.values())
    return smallest, largest


# In[416]:

smallest, largest = smallest_largest_coefficient(model_coefficients)
print("smallest power 1 coefficient")
print(smallest)


# In[417]:

print("largest power 1 coefficient")
print(largest)


# ## Ridge regression comes to rescue

# Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)
# 
# With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.

# In[418]:

penalty_2 = 1e5
assert penalty_2 == 10**5
models_2 = get_models(penalty=penalty_2)


# In[419]:

model_2_coefficients = get_coefficients(models_2)
for name, coefficients in model_2_coefficients.items():
    print(name)
    print(coefficients)


# In[420]:

plot_models(models_2, penalty_2, palette='husl')


# These curves should vary a lot less, now that you applied a high degree of regularization.
# 
# ***QUIZ QUESTION:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# In[421]:

smallest_largest_coefficient(model_2_coefficients)


# ## Selecting an L2 penalty via cross-validation

# Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.
# 
# We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:
# 
# Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
# Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
# ...<br>
# Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# 
# After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that all observations are used for both training and validation, as we iterate over segments of data. 
# 
# To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. GraphLab Create has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use `seed=1` to get consistent answers.)

# In[422]:

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)


# Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

# With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.

# In[423]:

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)


# Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.

# In[424]:

train_valid_shuffled[0:10] # rows 0 to 9


# Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
# Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.

# In[425]:

validation4 = train_valid_shuffled[5818:7758]


# To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $536,234.

# In[426]:

actual = int(round(validation4['price'].mean(), 0))
expected = 536234
assert actual == expected, "actual - expected = {0}".format(actual-expected)


# After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has an `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.

# In[427]:

n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print first_two.append(last_two)


# Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.

# In[428]:

train4 = train_valid_shuffled[0:5818]
train4 = train4.append(train_valid_shuffled[7758:n])


# To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $539,450.

# In[429]:

actual = int(round(train4['price'].mean(), 0))
expected = 539450
assert actual == expected


# Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.
# 
# * For each i in [0, 1, ..., k-1]:
#   * Compute starting and ending indices of segment i and call 'start' and 'end'
#   * Form validation set by taking a slice (start:end+1) from the data.
#   * Form training set by appending slice (end+1:n) to the end of slice (0:start).
#   * Train a linear model using training set just formed, with a given l2_penalty
#   * Compute validation error using validation set just formed

# In[430]:

DataSets = namedtuple('DataSets', 'validation training poly_data features'.split())
DataSlices = namedtuple('DataSlices', 'validation training'.split())


# In[431]:

def get_slices(fold, k, raw_data, verbose=False):
    """
    :param:
     - `fold`: the current fold
     - `k`: number of folds
     - `verbose`: print slice start and end if True
    :return: DataSlices tuple
    """
    # setup the slice indices
    total = len(raw_data)
    start = (total * fold)/k
    end = (total * (fold + 1)/k) - 1
    
    if verbose:
        print('start: {0}'.format(start))
        print('end: {0}'.format(end))

    # get the validation and training sets
    if start == 0:
        training = raw_data[end + 1: total]
    else:
        training = raw_data[0:start].append(raw_data[end + 1: total])
    validation = raw_data[start:end + 1]
    return DataSlices(validation=validation, training=training)


# In[432]:

test_data = graphlab.SFrame({"x":range(100)})
for i in range(10):
    output = get_slices(i, 10, test_data)
    assert len(output.validation) == 10, len(output.validation)
    assert output.validation['x'][0] == i * 10, output.validation
    assert len(output.training) == 90, "i: {0} first_validation: {1}".format(i, output.validation[0])


# In[433]:

def get_data_sets(fold, k, raw_data, verbose=False):
    """
    :param:
     - `fold`: the current fold
     - `k`: total number of folds
     - `raw_data`: data to split up
     - `verbose`: if true, print some output as you go

    :return: DataSets tuple
    """
    data_slices = get_slices(fold=fold, k=k, raw_data=raw_data, verbose=verbose)
    # create the 15-degree polynomial data set
    poly_data = polynomial_sframe(data_slices.training[Constants.predictor], 15)
    features = poly_data.column_names()
    poly_data[Constants.target] = data_slices.training[Constants.target]
    
    return DataSets(validation=data_slices.validation,
                    training=data_slices.training,
                    poly_data=poly_data,
                    features=features)


# In[497]:

def get_model(poly_data, features, penalty, verbose=False):
    if verbose:
        print('model penalty: {0}'.format(penalty))
    return graphlab.linear_regression.create(poly_data,
                                             target=Constants.target,
                                             features=features,
                                             validation_set=None,
                                             l2_penalty=penalty,
                                             verbose=False)


# In[524]:

def get_error(fold, k, raw_data, penalty, data_sets=None, verbose=False):
    import pudb; pudb.set_trace()
    model = get_model(data_sets.poly_data, data_sets.features, penalty, verbose=verbose)
    if verbose:
        print(model.get('coefficients'))
    predictions = model.predict(data_sets.validation)
    residuals = predictions - data_sets.validation[Constants.target]
    return (residuals**2).sum()


def k_fold_cross_validation(k, l2_penalty, data, output_name, data_sets=None, verbose=False):
    errors = []
    for fold, data_set in enumerate(data_sets):
        error = get_error(fold=fold, k=k,
                          raw_data=data, data_sets=data_set, penalty=l2_penalty,
                          verbose=verbose)
        if verbose:
            print("k-fold ({0}) error: {1}".format(fold, error))
        errors.append(error)
    return sum(errors)/float(len(errors))
        


# In[486]:

training_sorted = train_valid_shuffled.sort([Constants.predictor, Constants.target])
for penalty in numpy.logspace(1, 7, num=13):
    figure = plot.figure()
    axe = figure.gca()

    data_sets = get_data_sets(fold=0, k=10,
                              raw_data=training_sorted,
                              verbose=False)
    
    model = get_model(data_sets.poly_data, data_sets.features, penalty)
    x = data_sets.poly_data[Constants.base_polynomial]
    y = model.predict(data_sets.poly_data)

    lines = axe.plot(x, data_sets.training[Constants.target], '.', label='data')
    lines = axe.plot(x, y, '-', label='model')
    legend = axe.legend(loc="upper left")
    title = axe.set_title("Sq Ft Living vs Price (degree 15, penalty {0}".format(penalty))
    label = axe.set_xlabel("Living Area (Sq Ft)")
    label = axe.set_ylabel("Price ($)")


# In[525]:

errors = []
folds = 2
ten_data_sets = [get_data_sets(fold, k=folds, raw_data=train_valid_shuffled)
                     for fold in range(folds)]
#for penalty in numpy.logspace(1, 7, num=13):

for penalty in [1e-5, 1e7]:
    error = k_fold_cross_validation(k=folds, l2_penalty=penalty,
                                    data=train_valid_shuffled,
                                    data_sets=ten_data_sets,
                                    output_name='what', verbose=True)
    print(penalty, error)
    errors.append(error)


# In[515]:

print(min(errors))
print(max(errors))
print(min(errors) == max(errors))
print(max(errors) - min(errors))


# Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
# * We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
# * For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
#     * Run 10-fold cross-validation with `l2_penalty`
# * Report which L2 penalty produced the lowest average validation error.
# 
# Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!

# In[ ]:




# ***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

# You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  

# In[ ]:

# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.



# Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`.  This way, your final model will be trained on the entire dataset.

# In[ ]:




# ***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***

# In[ ]:



