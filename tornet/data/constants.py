"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
Constants related to dataset
"""
import numpy as np

# List all potential input variables
ALL_VARIABLES=['DBZ',
               'VEL',
               'KDP',
               'RHOHV',
               'ZDR',
               'WIDTH'
            ]

# Provides a typical min-max range for each variable (but not exact)
# Used for normalizing in a NN
CHANNEL_MIN_MAX = {
    'DBZ': [-20.,60.],
    'VEL': [-60.,60.],
    'KDP': [-2.,5.],
    'RHOHV': [0.2, 1.04],
    'ZDR': [-1.,8.],
    'WIDTH':[0.,9.],
}

MAIDS_VARIABLES = [
    'madis_atmospheric_pressure',
    'madis_wind_direction',
    'madis_wind_speed',
    'madis_wind_gust_speed',
    'madis_relative_humidity',
    'madis_temperature',
    'madis_temperature_dew_point'
]

MADIS_MIN_MAX = [
    [90000., 110000.],   # madis_atmospheric_pressure
    [0., 360.],          # madis_wind_direction
    [0., 50.],           # madis_wind_speed
    [0., 50.],           # madis_wind_gust_speed
    [0., 100.],          # madis_relative_humidity
    [233.15, 323.15],         # madis_temperature
    [233.15, 323.15]         # madis_temperature_dew_point
]














