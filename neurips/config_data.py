from dataclasses import dataclass

@dataclass
class Paths:
  model_dir : str
  base_dir : str
  raw_dir : str
  raw_save_dir: str
  real_mask_dir : str
  binary_mask_dir : str
  binary_erode_mask_dir : str

@dataclass
class Params:

    pattern: str
    num_channels: int
    patch_x: int 
    patch_y: int 

@dataclass
class NeurIPSDataConfig:

     paths: Paths
     params: Params
    