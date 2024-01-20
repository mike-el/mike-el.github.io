```python
# Summary code to do the above. Ie load, explore and visualise.
""" Read in Hadley sea surface temperature anomaly data in NetCDF format. It spans 1850 - 2023.
Visualise the latest 2023 anomaly global picture using the Iris package in Python.
"""
import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import datetime
from iris.time import PartialDateTime

# https://www.metoffice.gov.uk/hadobs/hadsst4/data/netcdf/HadSST.4.0.1.0_median.nc')
anoms = iris.load_cube("C:/Users/User/Documents/Mike/2023_job_applications/python_training_2023/HadSST.4.0.1.0_median.nc")

print(anoms) # See the cube metadata.

# now get time coordinate information:
coord = anoms.coord('time')
print('time')
print(coord.points)
print(coord.units)

# Find the date time points.

print('All times :\n' + str(anoms.coord('time'))) # first one is 1850-01-16 12:00:00. Last is 2023-12-16 12:00:00.

# Extract the final slice. Use a function to select the slice inbetween 2 dates.

pdt1 = PartialDateTime(year=2023, month=12, day=14)
pdt2 = PartialDateTime(year=2023, month=12, day=18)
last_daterange = iris.Constraint(
    time=lambda cell: pdt1 <= cell.point < pdt2)
last_slice = anoms.extract(last_daterange) # extract the first slice using date constraints.
print(last_slice)

# Finally, plot it as filled contours with a brewer colormap.

qplt.contourf(last_slice, brewer_cmap.N, cmap=brewer_cmap)

# Add coastlines to the map created by contourf.
plt.gca().coastlines()

plt.show()
```
