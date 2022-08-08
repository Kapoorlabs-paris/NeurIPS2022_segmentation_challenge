from dataclasses import dataclass

@dataclass
class Params:
    n_tiles_x: int
    n_tiles_y: int 
    min_size : int 
    min_size_mask : int 
    max_size : int 
    dounet : bool 
    seedpool : bool 
    slice_merge : bool 
    UseProbability : bool 
    donormalize : bool 
    lower_perc : float  
    upper_perc: float 
    RGB : bool 
    axes : str 

@dataclass
class Files:
  star_model_name : str 
  unet_model_name : str

@dataclass
class Paths:
  model_dir : str 
  image_dir : str 
  save_dir : str  


@dataclass
class NeurIPSPredictConfig:

     paths: Paths
     files: Files
     params: Params   

