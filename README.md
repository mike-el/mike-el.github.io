# mike-el.github.io

# Mike Elliott
## Data science portfolio

 - Weather and climate modelling
 - Artificial Intelligence: Natural Language Processing
 - XGBoost for predictive modelling

## Weather data cube
Using the Met Office IRIS package, open a cube that contains many years worth of the Hadley sea surface temperature anomaly data, explore the dimensions and their ranges, select the most recent time slice, then visualise the latest 2023 anomaly global picture using a color-brewer palette with coastlines displayed on a filled contour plot:

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/hadley_last_slice.jpg)

The data are in NetCDF format, which is a standard data format for climate data. The Hadley sea surface temperature anomaly data spans 1850 - 2023.

![Code to perform the tasks above](Weather_data_cube.md)

## Climate extremes
Using a long time series of multi-dimensional weather station data, load into a dataframe, remove missing numbers and non-numeric identifiers (eg estimated data), classify each month of the time series as an outlier or not based on the K nearest neighbours clustering algorithm, then use the seaborn package to visualise the multi-dimensional data set with points labelled orange if they are classed as an outlier or not.

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/oxford_scatterplot_outliers.png)

Our historic station data consists of:

 - Mean daily maximum temperature (tmax)
 - Mean daily minimum temperature (tmin)
 - Days of air frost (af)
 - Total rainfall (rain)
 - Total sunshine duration (sun) - This is only recorded from 1929 onwards.

The scatter plot matrix is really useful. It shows:
 - bimodal distributions for temperature and sunshine
 - strong correlations of temperature min and max
 - strong correlations of tmax with sun_hrs and air_frost_days
 - the outliers (orange points) are not just high and low values in one or two variables. Instead they can be seen to be scattered amongst all the dimensions, suggesting that the algorithm has successfully identified multidimensional outliers
 - the scatterplot of sun_hrs and rain_mm is perhaps the most interesting with extreme values existing in all the edges of the chart, except for low rain_mm.

![Code to perform the tasks above](Climate_extremes.md)

## Forecasting with a boosted decision tree: XGBoost algorithm
Using a standard "video games" data set that contains columns like: sales volumes; critic and user scores; year of release; genre.

The project objective I set myself was a hard one
 - 'Critic_Score','Critic_Count','User_Score','User_Count','Rating','Year_of_Release' - the candidate explanatory variables.
 - 'Global_Sales' - the variable we want to predict.

The data (after data munging) was like this, with 1 row per game:

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/xgboost_data_optimisation.png)

Use hold-out validation to assess the forecast accuracy. See a scatter plot below of predictions vs actual global sales:

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/xgboost_after_optimisation.png)

The correlation between the predicted global sales and actual was 0.38, ie medium strength.

Then I used a randomised search approach to quickly (it took 9s) optimise the XGBoost parameters and repeated the XGBoost forecast. The correlation had improved slightly to r = 0.41.

![](https://github.com/mike-el/mike-el.github.io/blob/main/images/xgboost_before_optimisation.png)
