## Regression

Carry out a Regression (supervised learning) task: do the columns like "critics_score" and "genre" predict "Sales" of a game?


```python
# Use a data frame "bb" that I have tidied up - lots of cleansing was required.

# Now build an XGBoost regression model to predict Global_Sales from the numeric parameters (exc sales).

# Load data into 2 separate df's.
X=bb[['Critic_Score','Critic_Count','User_Score','User_Count','Rating','Year_of_Release']] # The candidate explanatory variables.
y=bb['Global_Sales']  # what we want to predict.
X['User_Score'] = X['User_Score'].astype('float64')   # changes the type of a column. Strings are called "object". It was an object. xgb needs numeric columns.
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model to training data
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
print(model)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[26], line 6
          1 # Use a data frame "bb" that I have tidied up - lots of cleansing was required.
          2 
          3 # Now build an XGBoost regression model to predict Global_Sales from the numeric parameters (exc sales).
          4 
          5 # Load data into 2 separate df's.
    ----> 6 X=bb[['Critic_Score','Critic_Count','User_Score','User_Count','Rating','Year_of_Release']] # The candidate explanatory variables.
          7 y=bb['Global_Sales']  # what we want to predict.
          8 X['User_Score'] = X['User_Score'].astype('float64')   # changes the type of a column. Strings are called "object". It was an object. xgb needs numeric columns.
    

    NameError: name 'bb' is not defined



```python
# make predictions for test data
y_pred = model.predict(X_test)
# Now put the actual and predicted global sales from the test sample into a df so they can be plotted against each other.
delme = pd.DataFrame(y_test)
delme['pred'] = y_pred
#delme.head()
#sns.scatterplot(x='Global_Sales',y='pred',data=delme)  # a scatter plot.
sns.jointplot(x='Global_Sales',y='pred',data=delme,kind='reg',scatter_kws={'s':0.5})  # a scatter plot with a regression line on it. kind='kde' gives a 2D KDE to show point density.
```


```python
delme[['Global_Sales','pred']].corr() # Is there a correlation between length of the Job Title string and Salary? r=0.38
```

The predictions are not very accurate -as can be seen from the chart.

### optimise the XGBoost hyperparameters

Optimise the xgb parameters using a grid search - use a randomised search (faster than checking every possible combination - but less thorough).


```python
# a simple timer function - to know how long things take.
def timer(start_time=None):
    if not start_time:  # when the fcn is called with timer(None) then timing starts.
        start_time = datetime.now()
        return start_time
    elif start_time:  # # when the fcn is called with timer(start_time) then timing ends.
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

folds = 3  # 3 fold cross validation.
param_comb = 5

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001) # Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class. Test sets all contain the same distribution of classes, or as close as possible.

model1 = xgb.XGBRegressor() # The model to test is a XGB regressor.

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator = model1,
            param_distributions = params,
            cv=5, n_iter = 50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state = 42)  # Set a repeatable seed for the randomised search.
```


```python
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable

random_cv.fit(X_train, y_train)

timer(start_time) # timing ends here for "start_time" variable
```


```python
print('\n Best estimator:')
print(random_cv.best_estimator_)
results = pd.DataFrame(random_cv.cv_results_)
```


```python
print('\n Best hyperparameters:')
print(random_cv.best_params_)
"""
 Best hyperparameters:
{'subsample': 1.0, 'min_child_weight': 10, 'max_depth': 5, 'gamma': 5, 'colsample_bytree': 1.0}
"""
```


```python
results.head()   # See some of the cross validation results. It has 50 rows.
```


```python
reg = xgb.XGBRegressor(
        tree_method="hist", subsample= 1.0, min_child_weight= 10, max_depth= 5, gamma= 5, colsample_bytree= 1.0
    )  # Build an xgb model that is set to the best hyperparameters found by the random search.
```


```python
reg.fit(X_train, y_train)  # Fit, then make predictions for test data - using the best parameters.
y_pred = reg.predict(X_test)
# Now put the actual and predicted global sales from the test sample into a df so they can be plotted against each other.
delme = pd.DataFrame(y_test)
delme['pred'] = y_pred
sns.jointplot(x='Global_Sales',y='pred',data=delme,kind='reg',scatter_kws={'s':0.5})  # a scatter plot with a regression line on it. kind='kde' gives a 2D KDE to show point density.
```


```python
delme[['Global_Sales','pred']].corr() # Is there a correlation between length of the Job Title string and Salary? r=0.41 
# The correlation is higher (ie good) than before the parameter optimisation, so optimisation has helped.
"""
                Global_Sales 	pred
Global_Sales 	1.000000 	0.408069
pred 	        0.408069 	1.000000
"""
```


```python

```
