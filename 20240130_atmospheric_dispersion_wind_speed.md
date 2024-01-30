# Project: load wind data, then use it to model atmospheric dispersion with monte carlo simulation


```python
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
import numpy.random as rng # the numpy random generator. For use to add noise/turbulence to the wind speeds, using standard_normal().

import iris
import iris.plot as iplt
import iris.quickplot as qplt

infile = iris.sample_data_path("wind_speed_lake_victoria.pp")
uwind = iris.load_cube(infile, "x_wind")
vwind = iris.load_cube(infile, "y_wind")

# Create a cube containing the wind speed.
windspeed = (uwind**2 + vwind**2) ** 0.5
windspeed.rename("windspeed")

# Plot the wind speed as a contour plot.
qplt.contourf(windspeed, 20)

# Show the lake on the current axes.
lakes = cfeat.NaturalEarthFeature(
        "physical", "lakes", "50m", facecolor="none"
)
plt.gca().add_feature(lakes)

# Add arrows to show the wind vectors.
iplt.quiver(uwind, vwind, pivot="middle")

plt.title("Wind speed over Lake Victoria")
qplt.show()

# Normalise the data for uniform arrow size.
u_norm = uwind / windspeed
v_norm = vwind / windspeed

# Make a new figure for the normalised plot.
plt.figure()

qplt.contourf(windspeed, 20)
plt.gca().add_feature(lakes)
iplt.quiver(u_norm, v_norm, pivot="middle")

plt.title("Wind speed over Lake Victoria")
qplt.show()

#plt.savefig('wind_speed_lake_victoria.png',dpi=200)  # To save the figure. You specify the extension (eg jpg, png, bmp) and matplotlib is clever enough to figure out how to save it.
```


    
![png](output_1_0.png)
    



    
![png](output_1_1.png)
    



    <Figure size 640x480 with 0 Axes>



```python
print(uwind)
print(vwind)
```

    x_wind / (m s-1)                    (latitude: 14; longitude: 17)
        Dimension coordinates:
            latitude                             x              -
            longitude                            -              x
        Scalar coordinates:
            forecast_period             777960.0 hours, bound=(777600.0, 778320.0) hours
            forecast_reference_time     1859-12-01 00:00:00
            height                      10.0 m
            time                        1949-12-16 00:00:00, bound=(1949-12-01 00:00:00, 1950-01-01 00:00:00)
        Cell methods:
            0                           time: mean (interval: 1 hour)
        Attributes:
            STASH                       m01s03i225
            source                      'Data from Met Office Unified Model'
    y_wind / (m s-1)                    (latitude: 14; longitude: 17)
        Dimension coordinates:
            latitude                             x              -
            longitude                            -              x
        Scalar coordinates:
            forecast_period             777960.0 hours, bound=(777600.0, 778320.0) hours
            forecast_reference_time     1859-12-01 00:00:00
            height                      10.0 m
            time                        1949-12-16 00:00:00, bound=(1949-12-01 00:00:00, 1950-01-01 00:00:00)
        Cell methods:
            0                           time: mean (interval: 1 hour)
        Attributes:
            STASH                       m01s03i226
            source                      'Data from Met Office Unified Model'
    


```python
for coord in uwind.coords():
    print(coord.name())
```

    latitude
    longitude
    forecast_period
    forecast_reference_time
    height
    time
    


```python
coord = uwind.coord('latitude')
print(coord.points)
print(coord.bounds)
print(coord.units)
```

    [ 1.9799995   1.5399997   1.0999999   0.6600002   0.22000039 -0.21999931
     -0.65999913 -1.099999   -1.5399988  -1.9799986  -2.4199982  -2.8599982
     -3.2999978  -3.7399979 ]
    None
    degrees
    


```python
coord = uwind.coord('longitude')
print(coord.points)
print(coord.bounds)
print(coord.units)
```

    [389.26    389.7     390.14    390.58002 391.02002 391.46    391.9
     392.34    392.78    393.22    393.66    394.1     394.53998 394.97998
     395.41998 395.86    396.3    ]
    None
    degrees
    


```python
coord = uwind.coord('time')
print(coord.points)
print(coord.units)
```

    [-173160.]
    hours since 1970-01-01 00:00:00
    

This shows the wind data is a vector located at points given by lat/long coords, and only for a single time point.

I will convert the lat and long to a square grid with units of metres so I can apply the wind vectors and time.

I will interpolate the wind from its discrete points to a more continuous and closely spaced grid, to be able to identify the wind speed as the simulated particles flow through the grid.


```python
# The Haversine formula to calculate approximate distance in metres between 2 lat/lon points. It is approximate as it
# ignores the non-spherical earth curvature. However, for this exercise and the small scale of the example it is fine.

from math import cos, asin, sqrt, pi

def distance(lat1, lon1, lat2, lon2):
    r = 6371000 # metres.
    p = pi / 180

    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * r * asin(sqrt(a))
```


```python
# Considering the wind speed map as a rectangle with a top, bottom, left, right:
# The distance in metres along the bottom of the map. 773329 metres, 773 km. Sense check this result:
# This distance feels about right as the widest breadth of lake victoria is 240km.
distance( -3.73999786 , 389.26000977 , -3.73999786 , 396.22958801 )
```




    773329.2506490127




```python
# The distance in metres along the top of the map. 774544 m, 775 km. longitude. x axis.
distance( 1.92279957 , 389.26000977 , 1.92279957 , 396.22958801 )
```




    774544.8441968124




```python
# The distance in metres along the right of the map. 629674 m, 629 km. latitude. y axis.
distance( -3.73999786 , 389.26000977 , 1.92279957 , 389.26000977 )
```




    629674.3448318471




```python
# The distance in metres along the left of the map. 629674 m, 629 km.
distance( -3.73999786 , 396.22958801 , 1.92279957 , 396.22958801 )
```




    629674.3448318471



Therefore the average distance between the 100 wind points of the map is:
    - 7,740 m (in the y axis), 
    - 6,296 m in the x axis.

### Planning how I'll do the dispersion calcns:
 - will it help to have lots of smaller squares? 
 
No, for this proof of concept model I think it's good to stick with big squares (10km). I will interpolate to have 77 wind points in y axis and 63 in x axis. This will make a square grid with each square 10km.

 - how will I add turbulence/noise to get a plume?  

Within each square of the grid (ignoring turbulence) the particle will move in the direction of the wind vector and I can calculate how long it takes to leave that square based on the speed and direction, and which square it moves into next. So rather than advancing one time unit in each loop iteration - I will advance by the "time" it takes to cross 1 square at a time. Store the start and end coords of each particle and time step (ie as it crosses 1 square), then I can chart the track of each particle over time. I can show multiple tracks (ie multiple particles - which makes sense if I can introduce noise/turbulence.). 

 - How to introduce turbulence to the wind dirn and speed? 
 
The wind is actually in the format of x and y speed components of a vector so all I have to do is have a gaussian noise component that is multiplicative by the speed (ie scales with the speed). This will introduce randomness to direction and speed. For speed of simulations (if I find the simulation runs slowly), I could calculate the multiplier at the start of each run and apply the same values to each square in that run (ie for that particle).


```python
# interpolate the wind data to a 10km grid.

import numpy as np

sample_points = [('longitude', np.linspace(389.26, 396.3, 63 )),
                 ('latitude',  np.linspace(1.9799995, -3.7399979, 77))] # 77 points.
uwind2 = uwind.interpolate(sample_points, iris.analysis.Linear())
print(uwind2.summary(shorten=True))

vwind2 = vwind.interpolate(sample_points, iris.analysis.Linear())
print(vwind2.summary(shorten=True))
```

    x_wind / (m s-1)                    (latitude: 77; longitude: 63)
    y_wind / (m s-1)                    (latitude: 77; longitude: 63)
    


```python
# For the new interpolated wind data, plot it with arrows to show the wind vectors.
iplt.quiver(uwind2, vwind2, pivot="middle")
plt.gca().add_feature(lakes)

plt.title("Wind speed over Lake Victoria")
qplt.show()
```


    
![png](output_16_0.png)
    



```python
# Convert the cube format to a dataframe so the maths is simple.

import pandas
from iris.pandas import as_data_frame

xw = as_data_frame(uwind2, copy=True)
yw = as_data_frame(vwind2, copy=True)
```

    C:\Users\User\anaconda3\Lib\site-packages\iris\pandas.py:899: FutureWarning: You are using legacy 2-dimensional behaviour in'iris.pandas.as_data_frame()'. This will be removed in a futureversion of Iris. Please opt-in to the improved n-dimensional behaviour at your earliest convenience by setting: 'iris.FUTURE.pandas_ndim = True'. More info is in the documentation.
      warnings.warn(message, FutureWarning)
    


```python
# inspect the dataframes.

xw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>389.260000</th>
      <th>389.373548</th>
      <th>389.487097</th>
      <th>389.600645</th>
      <th>389.714194</th>
      <th>389.827742</th>
      <th>389.941290</th>
      <th>390.054839</th>
      <th>390.168387</th>
      <th>390.281935</th>
      <th>...</th>
      <th>395.278065</th>
      <th>395.391613</th>
      <th>395.505161</th>
      <th>395.618710</th>
      <th>395.732258</th>
      <th>395.845806</th>
      <th>395.959355</th>
      <th>396.072903</th>
      <th>396.186452</th>
      <th>396.300000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.979999</th>
      <td>-0.368164</td>
      <td>-0.245458</td>
      <td>-0.122718</td>
      <td>0.000021</td>
      <td>0.092052</td>
      <td>-0.030906</td>
      <td>-0.153898</td>
      <td>-0.276889</td>
      <td>-0.399187</td>
      <td>-0.519374</td>
      <td>...</td>
      <td>-1.765918</td>
      <td>-1.439121</td>
      <td>-1.300710</td>
      <td>-1.225100</td>
      <td>-1.149491</td>
      <td>-1.073902</td>
      <td>-1.070408</td>
      <td>-1.077212</td>
      <td>-1.084017</td>
      <td>-1.090820</td>
    </tr>
    <tr>
      <th>1.904736</th>
      <td>-0.289152</td>
      <td>-0.203339</td>
      <td>-0.117502</td>
      <td>-0.031666</td>
      <td>0.035768</td>
      <td>-0.025645</td>
      <td>-0.087075</td>
      <td>-0.148504</td>
      <td>-0.226124</td>
      <td>-0.352302</td>
      <td>...</td>
      <td>-1.718939</td>
      <td>-1.425716</td>
      <td>-1.281438</td>
      <td>-1.186817</td>
      <td>-1.092196</td>
      <td>-0.997600</td>
      <td>-1.002255</td>
      <td>-1.021088</td>
      <td>-1.039920</td>
      <td>-1.058748</td>
    </tr>
    <tr>
      <th>1.829473</th>
      <td>-0.210141</td>
      <td>-0.161220</td>
      <td>-0.112286</td>
      <td>-0.063352</td>
      <td>-0.020517</td>
      <td>-0.020384</td>
      <td>-0.020252</td>
      <td>-0.020119</td>
      <td>-0.053062</td>
      <td>-0.185230</td>
      <td>...</td>
      <td>-1.671960</td>
      <td>-1.412311</td>
      <td>-1.262167</td>
      <td>-1.148534</td>
      <td>-1.034901</td>
      <td>-0.921298</td>
      <td>-0.934102</td>
      <td>-0.964963</td>
      <td>-0.995823</td>
      <td>-1.026676</td>
    </tr>
    <tr>
      <th>1.754210</th>
      <td>-0.131129</td>
      <td>-0.119101</td>
      <td>-0.107070</td>
      <td>-0.095039</td>
      <td>-0.076801</td>
      <td>-0.015123</td>
      <td>0.046571</td>
      <td>0.108266</td>
      <td>0.120001</td>
      <td>-0.018158</td>
      <td>...</td>
      <td>-1.624981</td>
      <td>-1.398906</td>
      <td>-1.242895</td>
      <td>-1.110251</td>
      <td>-0.977606</td>
      <td>-0.844997</td>
      <td>-0.865949</td>
      <td>-0.908838</td>
      <td>-0.951726</td>
      <td>-0.994603</td>
    </tr>
    <tr>
      <th>1.678947</th>
      <td>-0.052118</td>
      <td>-0.076982</td>
      <td>-0.101854</td>
      <td>-0.126725</td>
      <td>-0.133086</td>
      <td>-0.009862</td>
      <td>0.113394</td>
      <td>0.236651</td>
      <td>0.293064</td>
      <td>0.148915</td>
      <td>...</td>
      <td>-1.578002</td>
      <td>-1.385502</td>
      <td>-1.223624</td>
      <td>-1.071967</td>
      <td>-0.920311</td>
      <td>-0.768695</td>
      <td>-0.797797</td>
      <td>-0.852713</td>
      <td>-0.907629</td>
      <td>-0.962531</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 63 columns</p>
</div>




```python
yw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>389.260000</th>
      <th>389.373548</th>
      <th>389.487097</th>
      <th>389.600645</th>
      <th>389.714194</th>
      <th>389.827742</th>
      <th>389.941290</th>
      <th>390.054839</th>
      <th>390.168387</th>
      <th>390.281935</th>
      <th>...</th>
      <th>395.278065</th>
      <th>395.391613</th>
      <th>395.505161</th>
      <th>395.618710</th>
      <th>395.732258</th>
      <th>395.845806</th>
      <th>395.959355</th>
      <th>396.072903</th>
      <th>396.186452</th>
      <th>396.300000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.979999</th>
      <td>-0.966797</td>
      <td>-0.958734</td>
      <td>-0.950669</td>
      <td>-0.942604</td>
      <td>-0.935893</td>
      <td>-0.938665</td>
      <td>-0.941437</td>
      <td>-0.944210</td>
      <td>-0.938100</td>
      <td>-0.905345</td>
      <td>...</td>
      <td>-0.321081</td>
      <td>-0.246247</td>
      <td>-0.153246</td>
      <td>-0.054198</td>
      <td>0.044851</td>
      <td>0.143872</td>
      <td>0.076416</td>
      <td>-0.014819</td>
      <td>-0.106055</td>
      <td>-0.197266</td>
    </tr>
    <tr>
      <th>1.904736</th>
      <td>-0.933555</td>
      <td>-0.916010</td>
      <td>-0.898461</td>
      <td>-0.880912</td>
      <td>-0.863866</td>
      <td>-0.850346</td>
      <td>-0.836822</td>
      <td>-0.823299</td>
      <td>-0.808544</td>
      <td>-0.790097</td>
      <td>...</td>
      <td>-0.332618</td>
      <td>-0.260716</td>
      <td>-0.171745</td>
      <td>-0.077094</td>
      <td>0.017557</td>
      <td>0.112182</td>
      <td>0.072771</td>
      <td>0.014213</td>
      <td>-0.044344</td>
      <td>-0.102886</td>
    </tr>
    <tr>
      <th>1.829473</th>
      <td>-0.900313</td>
      <td>-0.873287</td>
      <td>-0.846253</td>
      <td>-0.819220</td>
      <td>-0.791838</td>
      <td>-0.762027</td>
      <td>-0.732207</td>
      <td>-0.702388</td>
      <td>-0.678987</td>
      <td>-0.674850</td>
      <td>...</td>
      <td>-0.344156</td>
      <td>-0.275184</td>
      <td>-0.190245</td>
      <td>-0.099991</td>
      <td>-0.009737</td>
      <td>0.080492</td>
      <td>0.069126</td>
      <td>0.043246</td>
      <td>0.017366</td>
      <td>-0.008506</td>
    </tr>
    <tr>
      <th>1.754210</th>
      <td>-0.867072</td>
      <td>-0.830564</td>
      <td>-0.794046</td>
      <td>-0.757528</td>
      <td>-0.719810</td>
      <td>-0.673707</td>
      <td>-0.627592</td>
      <td>-0.581477</td>
      <td>-0.549431</td>
      <td>-0.559602</td>
      <td>...</td>
      <td>-0.355693</td>
      <td>-0.289652</td>
      <td>-0.208744</td>
      <td>-0.122887</td>
      <td>-0.037031</td>
      <td>0.048803</td>
      <td>0.065480</td>
      <td>0.072279</td>
      <td>0.079077</td>
      <td>0.085873</td>
    </tr>
    <tr>
      <th>1.678947</th>
      <td>-0.833830</td>
      <td>-0.787840</td>
      <td>-0.741838</td>
      <td>-0.695835</td>
      <td>-0.647783</td>
      <td>-0.585388</td>
      <td>-0.522977</td>
      <td>-0.460566</td>
      <td>-0.419874</td>
      <td>-0.444354</td>
      <td>...</td>
      <td>-0.367231</td>
      <td>-0.304121</td>
      <td>-0.227243</td>
      <td>-0.145784</td>
      <td>-0.064325</td>
      <td>0.017113</td>
      <td>0.061835</td>
      <td>0.101311</td>
      <td>0.140787</td>
      <td>0.180253</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 63 columns</p>
</div>



The time taken for a particle to cross a cube will depend on its trajectory within each grid square, ie its start location, speed and direction. That is where it enters the square and where it exits.




```python
xw.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>389.260000</th>
      <th>389.373548</th>
      <th>389.487097</th>
      <th>389.600645</th>
      <th>389.714194</th>
      <th>389.827742</th>
      <th>389.941290</th>
      <th>390.054839</th>
      <th>390.168387</th>
      <th>390.281935</th>
      <th>...</th>
      <th>395.278065</th>
      <th>395.391613</th>
      <th>395.505161</th>
      <th>395.618710</th>
      <th>395.732258</th>
      <th>395.845806</th>
      <th>395.959355</th>
      <th>396.072903</th>
      <th>396.186452</th>
      <th>396.300000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>...</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
      <td>77.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.206844</td>
      <td>0.220168</td>
      <td>0.233496</td>
      <td>0.246823</td>
      <td>0.258244</td>
      <td>0.256306</td>
      <td>0.254367</td>
      <td>0.252429</td>
      <td>0.243376</td>
      <td>0.212981</td>
      <td>...</td>
      <td>-0.908281</td>
      <td>-0.982247</td>
      <td>-0.976986</td>
      <td>-0.945318</td>
      <td>-0.913650</td>
      <td>-0.881991</td>
      <td>-0.927367</td>
      <td>-0.983746</td>
      <td>-1.040124</td>
      <td>-1.096488</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.201292</td>
      <td>0.178904</td>
      <td>0.190530</td>
      <td>0.231101</td>
      <td>0.261803</td>
      <td>0.130049</td>
      <td>0.143657</td>
      <td>0.282376</td>
      <td>0.375696</td>
      <td>0.284011</td>
      <td>...</td>
      <td>0.753470</td>
      <td>0.730197</td>
      <td>0.636207</td>
      <td>0.516795</td>
      <td>0.406817</td>
      <td>0.316295</td>
      <td>0.330586</td>
      <td>0.374339</td>
      <td>0.430955</td>
      <td>0.496029</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.368164</td>
      <td>-0.245458</td>
      <td>-0.122718</td>
      <td>-0.181116</td>
      <td>-0.230812</td>
      <td>-0.030906</td>
      <td>-0.153898</td>
      <td>-0.343603</td>
      <td>-0.588796</td>
      <td>-0.519374</td>
      <td>...</td>
      <td>-2.085975</td>
      <td>-2.222892</td>
      <td>-2.120099</td>
      <td>-1.925567</td>
      <td>-1.731035</td>
      <td>-1.536556</td>
      <td>-1.600756</td>
      <td>-1.716607</td>
      <td>-1.930277</td>
      <td>-2.143889</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.052375</td>
      <td>0.052351</td>
      <td>0.116682</td>
      <td>0.096458</td>
      <td>0.067994</td>
      <td>0.185667</td>
      <td>0.204721</td>
      <td>0.087884</td>
      <td>-0.026109</td>
      <td>0.002911</td>
      <td>...</td>
      <td>-1.620203</td>
      <td>-1.598957</td>
      <td>-1.438736</td>
      <td>-1.277025</td>
      <td>-1.149491</td>
      <td>-1.102723</td>
      <td>-1.099632</td>
      <td>-1.187244</td>
      <td>-1.297659</td>
      <td>-1.430921</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.249666</td>
      <td>0.268787</td>
      <td>0.260875</td>
      <td>0.223820</td>
      <td>0.188734</td>
      <td>0.259491</td>
      <td>0.241351</td>
      <td>0.230217</td>
      <td>0.231382</td>
      <td>0.232187</td>
      <td>...</td>
      <td>-0.777628</td>
      <td>-0.869535</td>
      <td>-0.871829</td>
      <td>-0.913583</td>
      <td>-0.863016</td>
      <td>-0.797342</td>
      <td>-0.878534</td>
      <td>-0.908838</td>
      <td>-0.951726</td>
      <td>-0.994603</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.373201</td>
      <td>0.359321</td>
      <td>0.388765</td>
      <td>0.442080</td>
      <td>0.463938</td>
      <td>0.371568</td>
      <td>0.350918</td>
      <td>0.480251</td>
      <td>0.554073</td>
      <td>0.416789</td>
      <td>...</td>
      <td>-0.178504</td>
      <td>-0.326367</td>
      <td>-0.436140</td>
      <td>-0.494290</td>
      <td>-0.576379</td>
      <td>-0.629884</td>
      <td>-0.683308</td>
      <td>-0.730506</td>
      <td>-0.774884</td>
      <td>-0.799753</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.487562</td>
      <td>0.455127</td>
      <td>0.507629</td>
      <td>0.726637</td>
      <td>0.867811</td>
      <td>0.464079</td>
      <td>0.539276</td>
      <td>0.796531</td>
      <td>0.929184</td>
      <td>0.687966</td>
      <td>...</td>
      <td>0.212212</td>
      <td>0.164472</td>
      <td>-0.002564</td>
      <td>-0.209346</td>
      <td>-0.294626</td>
      <td>-0.374750</td>
      <td>-0.355214</td>
      <td>-0.321442</td>
      <td>-0.287669</td>
      <td>-0.253906</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 63 columns</p>
</div>




```python
# Replace the lat lon degree names for each row and column with integer indices.
xw2 = xw.reset_index()
xw2 = xw2.T.reset_index() # It has to be transposed for reset_index to function as it works on rows.
xw2 = xw2.T # Undo the transpose operation.
xw2 = xw2.iloc[1:78 , 1:64] # Remove the unwanted old row/col names. 77 rows, 63 columns.
xw2.head()

yw2 = yw.reset_index() # Now do the y wind speed.
yw2 = yw2.T.reset_index() # It has to be transposed for reset_index to function as it works on rows.
yw2 = yw2.T # Undo the transpose operation.
yw2 = yw2.iloc[1:78 , 1:64] # Remove the unwanted old row/col names. 77 rows, 63 columns.
```


```python
# Set the start conditions and constants for the model. x and y and model origin are relative to (0,0) being the starting square.

lim = 10000 # 10 km boundary to the x and y axis of each grid square. 77 rows, 63 columns, each square is 10km.
delta = 500 # an offset used to try and prevent a particle getting stuck in ping-pong between 2 squares.
track_steps = 200 # A limit for now many time steps each particle will get tracked for, before advancing to the next particle.
turbulence_scale = 1.5 # to scale the random Normal (sd = 1, mean = 0) multiplicative factor applied to the wind speed in each square.
starting_square = 32 # the number of squares from the left where the plume of particles is first emitted.
xw2.columns = list(range(-starting_square, 63-starting_square)) # rename the columns in xw2 to match the starting square. There are 63 columns in the grid.
yw2.columns = list(range(-starting_square, 63-starting_square)) # rename the columns in yw2 to match the starting square. There are 63 columns in the grid.
x_start_grid = 0 # starting square, counting from left. Use the xw2 dataframe index.
y_start_grid = 0 # starting square, counting down from top. Use the xw2 dataframe index.
df = pandas.DataFrame(columns = ['x','y','x_grid','y_grid','next_grid','time_step','cumulative_time','x_next','y_next', 'particle']) 
# create an empty dataframe to hold the particle global coordinates, units metres.
# x and y are the global coordinates (relative to the origin square within the overall grid).
# x_grid, y_grid are the dataframe cell indices for the square that the particle has just entered.
# next_grid is the relative location of the next square the particle will enter as it leaves this square. a=above, b=right, c=below, d=left.
# x_next and y_next are the starting coords within the next square (ie relative to the bottom left corner of that square) the particle enters.
# particle tracks which particle is being tracked. It increments when 1 track is completed (ie reaches a boundary).
#df.loc[0] = (xstart, ystart, x_start_grid, y_start_grid, 'c', 0, 0, lim/2, lim, 0) # initialise the first row.
#df.head()

data = {'next_gridd':['a','b','c','d'], # 1st column of dictionary. Defining what change in the grid squares occurs.
        'dx':[0,1,0,-1], # the x change.
        'dy':[1,0,-1,0]} # the y change.
gridd = pandas.DataFrame(data)  # Convert to a df. It has 3 columns, 4 rows.

```


```python
# Now the logic for tracking the particle's path.
row_counter = -1 # initiate the overall row counter that is the current one being written to in df. it increases by 1 for each time step.

for t in range(0,5): # range(0): #  integers from 0 to 4. Track 5 particles.
    # Initiate variables for each particle:
    xstart = x_start_grid*lim+lim/2 # start the plume of particles about halfway along the top of the model boundary, 32 squares from left. Plus half a square (5000m) so it's clear which the entry square will be.
    ystart = y_start_grid*lim # start the plume at the top of the model boundary, this is row 0. 
    boundary = 0
    cumu_time = 0 # reset the time counter.
    df.loc[row_counter] = (xstart, ystart, x_start_grid, y_start_grid, 'c', 0, 0, lim/2, lim, t) # initialise the first row.  
    tt = 0 # initiate a counter for the time step, for each particle (ie gets reset when another particle is tracked).
    while boundary == 0: # the check of whether a particle has reached one of the overall boundaries, and will stop being tracked.
        row_counter += 1 # Increment the row counter for df.
        tt += 1 # increment the time step counter for this particle.
        x = df.iloc[row_counter,0]
        y = df.iloc[row_counter,1]
        ng = df.iloc[row_counter,4] # get the code for the grid change.
        xg = df.iloc[row_counter,2] # get the x grid square.
        yg = df.iloc[row_counter,3] # get the y grid square.
        dx1 = gridd[gridd['next_gridd']==ng]['dx'] # get the grid change.
        dy1 = gridd[gridd['next_gridd']==ng]['dy']
        cumu_time = df.iloc[row_counter,6]
        x_next = df.iloc[row_counter,7]
        y_next = df.iloc[row_counter,8]
        nx, ny = turbulence_scale * rng.normal(size = 2, loc = 1, scale = 0.3) # 2 multiplicative factors to apply to the x & y wind speed in each square, mean = 1, sd = 0.1.
        xv = xw2.loc[yg,xg] * nx # x component of the wind speed from the xw2 dataframe - with turbulence. .loc[] expects rows, then columns.
        yv = yw2.loc[yg,xg] * ny # y component of the wind speed from the yw2 dataframe. .loc[] expects rows, then columns.

        # Calculations for the time taken for the particle to traverse the sqaure in this time step.
        if (xv < 0):
            tx = (0 - x_next)/xv # The time taken to reach the square boundary, x = 0.
        else: # (xv >= 0)
            tx = (lim - x_next)/xv # The time taken to reach the square boundary, x = lim.
        if (yv < 0):
            ty = (0 - y_next)/yv # The time taken to reach the square boundary, y = 0.
        else: # (yv >= 0)
            ty = (lim - y_next)/yv # The time taken to reach the square boundary, y = lim.

        # Calculations for the coordinates of the particle at the end of this time step.
        if (tx < ty and xv < 0): # the particle has reached the left boundary of this cell. This is the limiting constraint.
            x_next = lim - delta # the starting x coord within the next cell will be lim.
            y_next = y_next + (tx * yv) # the starting y coord within the next cell.
            next_grid = 'd' # In the next time step, the particle moves into the cell to the left.
            xg = xg - 1 # move to the square to the left.
            yg = yg # unchanged.
            #print(1)
        elif (tx < ty and xv >= 0): # the particle has reached the right boundary of this cell.
            x_next = delta
            y_next = y_next + (tx * yv)
            next_grid = 'b'
            xg = xg + 1
            yg = yg
            #print(2)
        elif (tx >= ty and yv < 0): # the particle has reached the bottom boundary of this cell.
            x_next = x_next + (ty * xv)
            y_next = lim - delta
            next_grid = 'c'
            xg = xg
            yg = yg + 1 # Note: because of the dataframe indexing, + 1 is moving down the map.
            #print(3, x_next, y_next, next_grid)
        else: # (tx < ty and xv >= 0): # the particle has reached the top boundary of this cell.
            x_next = x_next + (ty * xv)
            y_next = delta
            next_grid = 'a'
            xg = xg
            yg = yg - 1
            #print(4)

        # Now update the values of df ready for the next time step.
        x = x + min(tx, ty) * xv # update the overall/global coordinates. min(tx, ty) chooses whichever is the constraint of tx or ty.
        y = y + min(tx, ty) * yv
        #print (x,y, xg, yg, xv, yv, tx, ty, x_next, y_next)
        df.loc[row_counter] = (int(x), int(y), int(xg), int(yg), next_grid, tt, int(cumu_time + min(tx, ty)), int(x_next), int(y_next), int(t)) # save the values for the next time step.
        df.loc[0] = (xstart, ystart, x_start_grid, y_start_grid, 'c', 0, 0, lim/2, lim, 0) # initialise the first row.

        #df = pandas.DataFrame(columns = ['x','y','x_grid','y_grid','next_grid','time_step','cumulative_time','x_next','y_next', 'particle']) 
        if int(xg) == -32 or int(xg) == 30 or int(yg) == 0 or int(yg) == 76 or int(tt) == track_steps or int(row_counter) == 5000: 
            boundary = 1 # a boundary has been reached. advance to the next particle. Check tt and row_counter to limit the simulation run time.
        else: 
            boundary = 0 # keep the while loop operating.
        # end of the while loop that tracks 1 particle.
    # end of the for loop that increments to the next particle.
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>x_grid</th>
      <th>y_grid</th>
      <th>next_grid</th>
      <th>time_step</th>
      <th>cumulative_time</th>
      <th>x_next</th>
      <th>y_next</th>
      <th>particle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>5000.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c</td>
      <td>0</td>
      <td>0</td>
      <td>5000.0</td>
      <td>10000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>c</td>
      <td>0</td>
      <td>0</td>
      <td>5000.0</td>
      <td>10000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>573.0</td>
      <td>-10000</td>
      <td>0</td>
      <td>1</td>
      <td>c</td>
      <td>1</td>
      <td>9997</td>
      <td>573.0</td>
      <td>9500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>-10884</td>
      <td>-1</td>
      <td>1</td>
      <td>d</td>
      <td>2</td>
      <td>10458</td>
      <td>9500.0</td>
      <td>8615</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4694.0</td>
      <td>-19499</td>
      <td>-1</td>
      <td>2</td>
      <td>c</td>
      <td>3</td>
      <td>14930</td>
      <td>4805.0</td>
      <td>9500</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualise the particle tracks.
fig, ax = plt.subplots()
plt.style.use('default')
plt.plot(df[df['particle']==0]['x'],df[df['particle']==0]['y'], color='blue', linewidth=0.3, label='0')
plt.plot(df[df['particle']==1]['x'],df[df['particle']==1]['y'], color='red', linewidth=0.3, label='1')
plt.plot(df[df['particle']==2]['x'],df[df['particle']==2]['y'], color='gold', linewidth=0.3, label='2')
plt.plot(df[df['particle']==3]['x'],df[df['particle']==3]['y'], color='grey', linewidth=0.3, label='3')
plt.plot(df[df['particle']==4]['x'],df[df['particle']==4]['y'], color='black', linewidth=0.3, label='4')
plt.title('Particle tracks due to wind', fontsize=15)
plt.legend(title="particle",loc=3, fontsize='small', fancybox=True)
plt.xlabel('x change from start (m)', fontsize=13)
plt.ylabel('y change from start (m)', fontsize=13)
plt.show()
#plt.savefig('atmospheric_dispersion.png',dpi=200)  # To save the figure. You specify the extension (eg jpg, png, bmp) and matplotlib is clever enough to figure out how to save it.

```


    
![png](output_25_0.png)
    



```python

```
