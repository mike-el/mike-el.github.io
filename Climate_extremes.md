```python
"""
Project: classify extreme weather values.

Load Oxford weather station data into a data frame and use outlier detection to classify months as outliers or not - multidimensional cluster detection. Could try coming up with a predictive algorithm for outliers too - ie forecast when they will occur. However with data that is only monthly this has limited possibilities.

https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/oxforddata.txt

Monthly data are available for a selection of long-running historic stations. The series typically range from 50 to more than 100 years in length.
"""

import pandas as pd # Load the pandas library.
import numpy as np # Load the numpy library.
import seaborn as sns # load a nice plotting library
import matplotlib.pyplot as plt
from pyod.models.knn import KNN # the K nearest neighbours algorithm. (for outlier detection).

df=pd.read_csv('C:/Users/user/Documents/Mike/2023_job_applications/python_training_2023/oxford_weather_station.csv')  # Load a csv from a folder into a dataframe using the pandas function read_csv().
df.columns  # Print out what the columns are.
df.head()

df.info() # year and month columns are integers, the rest of the columns are strings because of the * and --- characters they contain.
# Next remove these special characters.

# MEXT: strip out the --- and * (missing data and estimated values - replace missing with NA).

for (index, colname) in enumerate(df.iloc[:,2:7]):
    df[colname] = df[colname].str.replace("*","")
    df[colname] = df[colname].replace("---",np.NaN)
    
df.tail()
df.head()

# change the types of columns from "object" to float.
df[['tmax', 'tmin', 'air_frost_days', 'rain_mm', 'sun_hrs']] = df[['tmax', 'tmin', 'air_frost_days', 'rain_mm', 'sun_hrs']].apply(pd.to_numeric)
df.info()

"""
Outlier detection using unsupervised learning: KNN clustering

from pyod.models.knn import KNN # the K nearest neighbours algorithm. 

The Key KNN parameters are

    contamination = proportion of outlier that I expect are in the data, eg 0.1 = 10%.
    method = how to calculate outliers, options are largest, mean.
    metric = how the distance between the n_neighbours is calc'd, eg 'minkowski' (default - a hybrid of euclidean and manhattan), euclidean (or l2), manhattan (or l1), 'mahalanobis' (use for unimodal data - it effectively normalises the data so scales/units aren't important).
    n_neighbours = number of neighbours to use when building clusters. Default is 5. Ideally, you will want to run for different KNN models with varying values of n_neighbours and compare the results to determine the optimal number of n_neighbors.

KNN cannot handle categorical data or missing data.
"""

df = df.dropna()  # returns a dataframe where any row with a missing value is removed.
df.info # 1140 rows (it was 2052). Because the sun_hrs column is not populated until 1929.

# Instantiate KNN with the updated parameters and then train (fit) the model using the dataframe that I cleaned up missing values:
knn = KNN(contamination=0.05,method='mean',n_neighbors=5)  # Assume that 5% of months are outliers.
knn.fit(df)  # Now fit a KNN model to the cleaned up df dataframe.

# The predict method will generate binary labels, either 1 or 0, for each data point. A value of 1 indicates an outlier. Store the results in a pandas Series and filter the predicted Series to only show the outlier values
predicted = pd.Series(knn.predict(df),index=df.index) 

knn_scores = knn.decision_scores_  # extract the decision scores from the KNN fit.

abc = pd.DataFrame(knn_scores) # convert to a df.

abc.index = df.index # change the index on abc so I can tell which row matches df.

abc = abc.rename(columns = {0:'scores'})  # to rename the column.

# Since all the data points are scored, PyOD will determine a threshold to limit the number of outliers returned. The threshold value depends on the contamination value you provided earlier (the proportion of outliers you suspect). The higher the contamination value, the lower the threshold, and hence more outliers are returned. 
knn.threshold_ # get the threshold value using the threshold_ attribute

# Now find the rows in abc which exceed the threshold score, ie have been classified as outliers.
abc[abc['scores'] >= knn.threshold_].sort_values('scores', ascending=False) # There are 57. 5% good.

ix = ( abc['scores'] > knn.threshold_ ) # create a mask. a boolean list depending on whether that row is above the threshold.
df['outlier'] = ix # add a column that contains the outlier classification.

sns.pairplot(data=df, hue='outlier')  # Use a pairplot to examine the multidimensional data.
plt.savefig("oxford_scatterplot_outliers.png")

"""
The scatter plot matrix is really useful. It shows
 - bimodal distributions for temperature and sunshine
 - strong correlations of temperature min and max
 - strong correlations of tmax with sun_hrs and air_frost_days
 - the outliers (orange points) are not just high and low values in one or two variables. Instead they can be seen to be scattered amongst all the dimensions, suggesting that the algorithm has successfully identified multidimensional outliers
 - the scatterplot of sun_hrs and rain_mm is perhaps the most interesting with extreme values existing in all the edges of the chart, except for low rain_mm.
"""
```
