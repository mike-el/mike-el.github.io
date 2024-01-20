# mike-el.github.io

# Mike Elliott
## Data science portfolio

 - Weather and climate modelling
 - Artificial Intelligence: Natural Language Processing
 - XGBoost for predictive modelling

## Weather data cube
Using the Met Office IRIS package, open a cube that contains many years worth of the Hadley sea surface temperature anomaly data, explore the dimensions and their ranges, select the most recent time slice, then visualise it using a color-brewer scale with coastlines displayed:

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/hadley_last_slice.jpg)

## Climate extremes
Using a long time series of multi-dimensional weather station data, load into a dataframe, remove missing numbers and non-numeric identifiers (eg estimated data), classify each month of the time series as an outlier or not based on the K nearest neighbours clustering algorithm, then use the seaborn package to visualise the multi-dimensional data set with points labelled orange if they are classed as an outlier or not.

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/oxford_scatterplot_outliers.png)

Our historic station data consists of:

 Mean daily maximum temperature (tmax)
 Mean daily minimum temperature (tmin)
 Days of air frost (af)
 Total rainfall (rain)
 Total sunshine duration (sun) - This is only recorded from 1929 onwards.

The scatter plot matrix is really useful. It shows
 - bimodal distributions for temperature and sunshine
 - strong correlations of temperature min and max
 - strong correlations of tmax with sun_hrs and air_frost_days
 - the outliers (orange points) are not just high and low values in one or two variables. Instead they can be seen to be scattered amongst all the dimensions, suggesting that the algorithm has successfully identified multidimensional outliers
 - the scatterplot of sun_hrs and rain_mm is perhaps the most interesting with extreme values existing in all the edges of the chart, except for low rain_mm.


