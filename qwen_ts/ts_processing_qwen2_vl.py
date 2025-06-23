import pandas as pd 

from typing import Optional, Union
from transformers.utils import TensorType

from sklearn.preprocessing import StandardScaler    
from transformers.image_processing_utils import BatchFeature

class Qwen2VLTSProcessor:
    def __init__(self,
                 features: str = 'S',
                 scale: bool = True,
                 patch_len: int = 16):
        self.features = features
        self.scale = scale
        self.patch_len = patch_len

    def __call__(
            self,     
            data_path: str = None,
            target: Optional[str] = None, 
            return_tensors: Optional[Union[str, TensorType]] = None):
        
        """
        Processes a time series CSV file and prepares it for model input.

        Args:
            data_path (`str`, *optional*):  
                Path to the CSV file containing the time series data.  
                The file should contain a `date` column and one or more feature columns.

            target (`str`, *optional*):  
                Name of the target column to be predicted.  
                If not specified, only features will be processed.

            return_tensors (`str` or `TensorType`, *optional*):  
                The type of tensors to return. Can be one of:  
                - `'pt'` or `TensorType.PYTORCH`: Return a batch of PyTorch tensors.  
                - `'tf'` or `TensorType.TENSORFLOW`: Return a batch of TensorFlow tensors.  
                - `'np'` or `TensorType.NUMPY`: Return a batch of NumPy arrays.  
                - `'jax'` or `TensorType.JAX`: Return a batch of JAX arrays.  
                - `None` (default): Return a list of NumPy arrays.

        Returns:
            Union[dict, BatchFeature]: Processed time series features.
        """
        
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(data_path)

        if target is not None:
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            cols = list(df_raw.columns)
            cols.remove(target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [target]]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = {"timeseries_values" : self.scaler.transform(df_data.values)}
        else:
            data = {"timeseries_values" : df_data.values}

        return BatchFeature(data=data, tensor_type=return_tensors)
