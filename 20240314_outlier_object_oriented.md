# Objective - using the object oriented programming (OOP) approach, carry out outlier detection in multidimendional data.

## Plan
- create a class to hold the data with __init__ . Instantiate any attributes like: name of file to load, directory path. Load libraries. Set attributes of the class.
  - pass the name and file path to the object when the __init__ is first invoked. It returns the dataframe as an attribute.
- write methods to process the dataframe (eg remove rows and columns, update missing values)
  - replace missing years for each name AND publisher combination with the min() by building a lookup table to hold this; then if a row still has NA then remove it.
  - Replace the Rating with a numerical score - as it is ordered.
  - remove non-numeric columns and then any rows with NA's.
  - Instantiate KNN with the updated parameters and then train (fit) the model using the dataframe b (that I cleaned up missing values); then fit it. An attribute holds the model object.
  - Use the predict method to generate binary labels, either 1 or 0, for each data point. Return the binary labels


```python
# Load the libraries required to load and feature engineer the dataset. Do this outside of the class (as otherwise they are only available inside the method they were declared in).
import pandas as pd # Load the pandas library.
import numpy as np # Load the numpy library.
import seaborn as sns # load a nice plotting library.
from pyod.models.knn import KNN # the K nearest neighbours clustering algorithm (to use for outlier detection).

class dataset:  # Define an object called dataset to hold our dataframe after it has been loaded.
    def __init__(self, filepath):  # When the model is first invoked it needs to be passed a file path.
        # Now load the dataset into an attribute:
        self.dframe = pd.read_csv(filepath) # Load a csv from a folder into a dataframe using the pandas function read_csv().
        self.knn = KNN()
        
    def feature_engineer(self):  # The feature engineering is pretty specific to this particular dataset, not generic.
        
        # replace missing years for each name AND publisher combination with the min() by building a lookup table to hold this; then if a row still has NA then remove it.
        a=self.dframe.groupby(['Name','Publisher']).agg({'Year_of_Release': min})  # This groups by the 'Name' and 'Publisher' columns and returns the minimum Year.
        a=a.reset_index() # a was a multilevel hierarchy. This returns it to a conventional df.
        # Now replace Year in df with the Year from a.
        b=pd.merge(self.dframe,a,how='inner',on=['Name','Publisher'])  # Merges 2 tables called df and a using an inner join, they both have columns called 'Name' and 'Publisher', used to join on. So this creates a new df with the columns of 'left' and then the columns of 'right' added to the rhs (except for the name and publisher column which was already in 'df'). The 'inner' can be replaced by outer, left or right. If the how=... is missed out then it defaults to an inner join.
        b.drop('Year_of_Release_x', axis=1, inplace=True) # Remove the old Year column.
        b.rename(columns={'Year_of_Release_y' : 'Year_of_Release'}, inplace=True) # Replace the column name.
        # Now remove any rows with NA in the Year column.
        b.dropna(axis=0, subset=['Year_of_Release'], inplace=True) # remove rows where there is NA in the Year column, update b.
        
        # prepare data for KNN cluster analysis (ie numeric vars without NA's)
        # remove non-numeric columns
        bb = b.drop(['Name','Platform','Publisher','Genre','Developer'], axis=1) # To remove columns. If inplace=True is not added the column is not permanently removed.
        bb = bb.dropna()  # returns a dataframe where any row with a missing value is removed.

        # Replace the Rating with a numerical score - as it is ordered.
        # Convert categorical data to numerical data using replace
        bb['Rating'] = bb['Rating'].replace({'E': 1, 'K-A': 1, 'E10+': 2, 'T': 3, 'M': 4, 'RP': 4, 'AO': 5})

        self.dframe = bb  # update the class attribute with the results of the feature engineering.
        
    def fit_knn(self, contamination_, method_, n_neighbors_):  # pass in some new attributes specific to this method.
        # Instantiate KNN with the updated parameters and then train (fit) the model using the dataframe b (that I cleaned up missing values):
        self.knn = KNN(contamination = contamination_, method = method_, n_neighbors = n_neighbors_)   # contamination hyperparameter = proportion of the data set I expect to classify as outliers, ie 3% here. 'mean' = distance method. The algorithm is defining the outlier distance (ie score) by comparing to its 20 nearest neighbours.
        self.knn.fit(self.dframe)  # Now fit a KNN model to the cleaned up dataframe. Store the KNN model as an attribute.
        #print(contamination_, method_, n_neighbors_)

    def check_outliers(self):
        # The predict method will generate binary labels, either 1 or 0, for each data point. A value of 1 indicates an outlier. Store the results in a pandas Series and filter the predicted Series to only show the outlier values
        predicted = pd.Series(self.knn.predict(self.dframe), index = self.dframe.index) 

        # Since all the data points are scored, KNN will determine a threshold to limit the number of outliers returned. The threshold value depends on the contamination value you provided earlier (the proportion of outliers you suspect). The higher the contamination value, the lower the threshold, and hence more outliers are returned. 
        # knn.threshold_ # get the threshold value using the threshold_ attribute. It is 43.
        # Now find the rows in abc which exceed the threshold score, ie have been classified as outliers.
        abc = pd.DataFrame(self.knn.decision_scores_) # convert to a df. Extract the decision scores from the KNN fit.
        abc = abc.rename(columns = {0:'scores'})  # to rename the column.
        #abc[abc['scores'] >= self.knn.threshold_].sort_values('scores', ascending=False) # There are 207.
        #n = int(len(self.dframe)*0.03) # The threshold score has found 207 outliers which is about 0.03 of the total number of rows (206), so is correct.


        # Visualise the outlier classification to see if it looks like it has worked successfully.
        ix = ( abc['scores'] > self.knn.threshold_ ) # create a mask. a boolean list depending on whether that row is above the threshold.
        self.dframe['outlier'] = ix # add a column that contains the outlier classification.
        sns.pairplot(data=self.dframe, hue='outlier')  # Use a pairplot to examine the multidimensional data.
        # This shows that User_Count is best for matching outliers - but Outliers are not only the games/platforms which are the most popular - so I think the approach is working to detect outliers.
        # The principle is I am using the KNN algorithm to compute a score (called local outlier factor) reflecting the degree of abnormality of the observations. It measures the local density deviation of a given data point with respect to its neighbors. The idea is to detect the samples that have a substantially lower density than their neighbors. The question is not, how isolated the sample is, but how isolated it is with respect to the surrounding neighborhood.
```


```python
# Load the dataframe.
df = dataset('C:/Users/Video_Games.csv') # Load a csv from a folder into a dataframe
df.dframe.head()
df.dframe.shape
```




    (16719, 16)




```python
# Carry out feature engineering.
df.feature_engineer()
df.dframe.shape
```




    (6871, 11)




```python
# Fit a KNN cluster model.
df.fit_knn(0.03, 'mean', 20)   # contamination hyperparameter = proportion of the data set I expect to classify as outliers, ie 3% here. 'mean' = distance method. The algorithm is defining the outlier distance (ie score) by comparing to its 20 nearest neighbours.
df.knn.threshold_ # get the threshold value using the threshold_ attribute. It is 43.
```




    42.96205979951985




```python
# Visualise the outlier classification to see if it looks like it has worked successfully.
df.check_outliers()
```




    2




```python

```
