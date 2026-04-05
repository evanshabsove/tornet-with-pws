"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
Tools to read tornado samples
"""
from typing import Dict, List, Callable
import datetime
import numpy as np
import xarray as xr

from tornet.data.constants import ALL_VARIABLES
import os
import pandas as pd

# Global MADIS data cache - loaded once at module level for efficiency
_MADIS_DATA_CACHE = None

def _load_madis_data(data_root):
    """
    Load MADIS features from CSV file once and cache in memory.
    Returns DataFrame indexed by (storm_id, timestamp) for fast lookup.
    """
    global _MADIS_DATA_CACHE
    
    if _MADIS_DATA_CACHE is not None:
        return _MADIS_DATA_CACHE
    
    csv_path = os.path.join(data_root, 'madis_features_clean.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MADIS CSV file not found at {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to string format matching netCDF time format
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(str)
    
    # Set multi-index for fast lookup by (storm_id, timestamp)
    df = df.set_index(['storm_id', 'timestamp'])
    
    # Sort index to avoid performance warnings
    df = df.sort_index()
    
    _MADIS_DATA_CACHE = df
    return df

def read_file(f: str,
              variables: List['str']=ALL_VARIABLES,
              n_frames:int=1,
              tilt_last:bool=True,
              use_madis_data:bool=False) -> Dict[str,np.ndarray]:
    """
    Extracts data from a single netcdf file

    Inputs:
    f: nc filename
    variables:  List of radar variables to load.  
                Default is all 6 variables ['DBZ','VEL','KDP','RHOHV','ZDR','WIDTH']
    n_frames:  number of frames to use. 1=last frame only.  No more than 4
               Default is 1.
    tilt_last:  If True (default), order of dimensions is left as [time,azimuth,range,tilt]
                If False, order is permuted to [time,tilt,azimuth,range]

    Returns:
    Dict containing data for each variable, along with several metadata fields.
    """
    
    data = {}
    with xr.open_dataset(f) as ds:

        # Load each radar variable
        for v in variables:
            data[v]=ds[v].values[-n_frames:,:,:,:]
        # Various numeric metadata
        data['range_folded_mask'] = ds['range_folded_mask'].values[-n_frames:,:,:,:].astype(np.float32) # only two channels for vel,width
        data['label'] = ds['frame_labels'].values[-n_frames:] # 1 if tornado, 0 otherwise
        data['category']=np.array([{'TOR':0,'NUL':1,'WRN':2}[ds.attrs['category']]],dtype=np.int64) # tornadic, null (random), or warning
        data['event_id']=np.array([int(ds.attrs['event_id'])],dtype=np.int64)
        data['ef_number']=np.array([int(ds.attrs['ef_number'])],dtype=np.int64)
        data['az_lower']=np.array(ds['azimuth_limits'].values[0:1])
        data['az_upper']=np.array(ds['azimuth_limits'].values[1:]) 
        data['rng_lower']=np.array(ds['range_limits'].values[0:1])
        data['rng_upper']=np.array(ds['range_limits'].values[1:])
        data['time']=(ds.time.values[-n_frames:].astype(np.int64)/1e9).astype(np.int64)

        if use_madis_data:
            # Get storm ID and timestamp
            storm_id = get_id_from_storm_event_url(ds.attrs['storm_event_url'])
            # Convert numpy datetime to pandas datetime
            timestamp_np = ds['time'].values[0]
            timestamp_dt = pd.to_datetime(timestamp_np)
            
            # Extract data_root from file path (goes up 3 levels from train/test/year/file.nc)
            data_root = os.path.dirname(os.path.dirname(os.path.dirname(f)))
            
            try:
                # Load MADIS data (cached after first call)
                madis_df = _load_madis_data(data_root)
                
                # Get all entries for this storm_id
                storm_id_int = int(storm_id)
                storm_data = madis_df.loc[storm_id_int]
                
                if storm_data.empty:
                    return None  # No MADIS data for this storm
                
               # Group by timestamp and average (handles multiple stations at same time)
                storm_data_agg = storm_data.groupby(level=0).mean(numeric_only=True)
                
                # Convert timestamps to datetime for comparison
                storm_timestamps = pd.to_datetime(storm_data_agg.index)
                
                # Find the temporally closest MADIS observation (within 10 minutes)
                time_diffs = abs(storm_timestamps - timestamp_dt)
                min_diff_idx = time_diffs.argmin()
                min_diff_seconds = time_diffs[min_diff_idx].total_seconds()
                
                # Only use MADIS data if within 10 minutes
                if min_diff_seconds > 600:  # 10 minutes = 600 seconds
                    return None  # No nearby MADIS observation
                
                # Get the row at the best matching timestamp
                best_timestamp = storm_timestamps[min_diff_idx]
                madis_row = storm_data_agg.loc[best_timestamp.strftime('%Y-%m-%d %H:%M:%S')]
                
                # Extract the 7 MADIS features in the correct order
                # CSV columns: pressure, wind_direction, wind_speed, wind_gust, relative_humidity, temperature, dewpoint
                # Model expects: pressure, wind_direction, wind_speed, wind_gust, relative_humidity, temperature, dewpoint
                madis_values = [
                    float(madis_row['pressure']),        # madis_atmospheric_pressure
                    float(madis_row['wind_direction']),  # madis_wind_direction
                    float(madis_row['wind_speed']),      # madis_wind_speed
                    float(madis_row['wind_gust']),       # madis_wind_gust_speed
                    float(madis_row['relative_humidity']), # madis_relative_humidity
                    float(madis_row['temperature']),     # madis_temperature
                    float(madis_row['dewpoint'])         # madis_temperature_dew_point
                ]
                
                # Check for missing values (NaN only - zeros are valid)
                if any(pd.isna(v) for v in madis_values):
                    return None  # Skip samples with missing MADIS data
                
                data['madis'] = np.array(madis_values, dtype=np.float32)
                
            except (KeyError, ValueError):
                # Storm ID or timestamp not found in MADIS data
                return None  # Skip samples without MADIS data

        # Store start/end times for tornado (Added in v1.1)
        if ds.attrs['ef_number']>=0 and ('tornado_start_time' in ds.attrs):
            start_time=datetime.datetime.strptime(ds.attrs['tornado_start_time'],'%Y-%m-%d %H:%M:%S')
            end_time=datetime.datetime.strptime(ds.attrs['tornado_end_time'],'%Y-%m-%d %H:%M:%S')
            epoch = datetime.datetime(1970,1,1)
            to_timestamp = lambda d: int((d - epoch).total_seconds())
            start_time=to_timestamp(start_time)
            end_time=to_timestamp(end_time)
        else:
            start_time=end_time=0
        data['tornado_start_time'] = np.array([start_time]).astype(np.int64)
        data['tornado_end_time'] = np.array([end_time]).astype(np.int64)

    # Fix for v1.0 of the data
    # Make sure final label is consistent with ef_number 
    data['label'][-1] = (data['ef_number'][0]>=0)
    
    if not tilt_last: 
        for v in variables+['range_folded_mask']:
            data[v]=np.transpose(data[v],(0,3,1,2))
        
    return data

def get_id_from_storm_event_url(storm_event_url):
    """
    Extracts the 'id' parameter value from the storm_event_url.
    Returns the id as a string, or None if not found.
    """
    import urllib.parse
    if storm_event_url:
        parsed = urllib.parse.urlparse(storm_event_url)
        query = urllib.parse.parse_qs(parsed.query)
        return query.get('id', [None])[0]
    return None


def query_catalog(data_root: str, 
                  data_type: str, 
                  years: list[int], 
                  random_state: int,
                  catalog: pd.DataFrame=None) -> list[str]:
    """Obtain file names that match criteria.
    If catalog is not provided, this loads and parses the 
    default catalog.

    Inputs:
    data_root: location of data
    data_type: train or test 
    years: list of years btwn 2013 - 2022 to draw data from
    random_state: random seed for shuffling files
    catalog:  Preloaded catalog, optional
    """
    if catalog is None:
        catalog_path = os.path.join(data_root,'catalog.csv')
        if not os.path.exists(catalog_path):
            raise RuntimeError('Unable to find catalog.csv at '+data_root)
        catalog = pd.read_csv(catalog_path,parse_dates=['start_time','end_time'])
    catalog = catalog[catalog['type']==data_type]
    catalog = catalog[catalog.start_time.dt.year.isin(years)]
    catalog = catalog.sample(frac=1, random_state=random_state) # shuffle file list
    file_list = [os.path.join(data_root,f) for f in catalog.filename]

    return file_list[0:100]

class TornadoDataLoader:
    """
    Tornado data loader class
    
    file_list:    list of TorNet filenames to load
    variables: list of TorNet variables to load (subset of ALL_VARIABLES)
    n_frames:  number of time frames to load (ending in last frame)
    shuffle:   If True, shuffles file_list before loading
    tilt_last:  If True (default), order of dimensions is left as [time,azimuth,range,tilt]
                If False, order is permuted to [time,tilt,azimuth,range]
                (if other dim orders are needed, use a transform)
    transform:  If provided, this callable is applied to transform each sample 
                before being returned

    """
    def __init__(self,
                 file_list:List[str],
                 variables: List['str']=ALL_VARIABLES,
                 n_frames:int=1,
                 shuffle:bool=False,
                 tilt_last:bool=True,
                 transform:Callable=None 
                 ):
        if shuffle:
            np.random.shuffle(file_list)
        self.file_list=file_list
        self.variables=variables
        self.n_frames=n_frames
        self.tilt_last=tilt_last
        self.current_file_index=0
        self.transform=transform
    def __iter__(self):
        self.current_file_index=0
        return self
    def __next__(self):
        if self.current_file_index<len(self):
            out = self[self.current_file_index]
            self.current_file_index+=1
            return out
        else:
            raise StopIteration
    def __getitem__(self,index:int):
        """
        Reads file at index
        """
        data = read_file(self.file_list[index],
                         variables=self.variables,
                         tilt_last=self.tilt_last,
                         n_frames=self.n_frames)

        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.file_list)


def get_dataloader(
    dataloader: str,
    data_root: str,
    years: list[int],
    data_type: str,
    batch_size: int,
    weights: Dict[str, float] = None,
    **kwargs,
):
    """Creates a dataloader for keras, torch or tensorflow

    Inputs:
    dataloader - str describing the dataloader to use. Valid options are: keras,
    tensorflow, tensorflow-tfds, torch and torch-tfds
    data_root - where data is located
    years - years to load btwn 2013 and 2022
    data_type - either train or test
    batch_size - size of batch
    weights - weights for different categories of sample

    weights is optional, if provided must be a dict of the form
      weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    where wN,w0,w1,w2,wW are numeric weights assigned to random,
    ef0, ef1, ef2+ and warnings samples, respectively.

    Returns:
    tf.data.Dataset, torch.utils.data.DataLoader or tornet.data.keras.loader.KerasDataLoader
    """

    # Argument validation
    valid_dataloaders = ["tensorflow", "tensorflow-tfds", "torch", "torch-tfds", "keras"]

    # Convert string to lower case
    dataloader = dataloader.lower()
    assert dataloader in valid_dataloaders, f"dataloader must be in {valid_dataloaders}!"

    from_tfds = False
    if "tfds" in dataloader:
        from_tfds = True

    if "tensorflow" in dataloader:
        import tensorflow as tf
        from tornet.data.tf.loader import make_tf_loader
        ds = make_tf_loader(data_root,data_type,years,batch_size,weights,from_tfds=from_tfds,**kwargs)

        data_opts = tf.data.Options()
        data_opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds=ds.with_options(data_opts)

    elif "torch" in dataloader:
        from tornet.data.torch.loader import make_torch_loader
        ds = make_torch_loader(data_root,data_type,years,batch_size,weights,from_tfds=from_tfds,**kwargs)
    else:
        from tornet.data.keras.loader import KerasDataLoader
        ds = KerasDataLoader(data_root=data_root,
                             data_type=data_type,
                             years=years,
                             batch_size=batch_size,
                             weights=weights,**kwargs)

    return ds
