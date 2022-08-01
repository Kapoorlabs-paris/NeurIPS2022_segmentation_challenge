from dataclasses import dataclass

@dataclass
class Paths:
  model_dir : str
  base_dir : str
  raw_dir : str
  real_mask_dir : str 
  binary_mask_dir : str
  binary_erode_mask_dir : str

@dataclass
class Files:

  npz_filename : str
  model_name : str


@dataclass
class Params:

    epoch: int
    learning_rate: float
    patch_x: int 
    patch_y: int 
    kern_size: int 
    n_patches_per_image: int 
    n_rays: int 
    startfilter: int 
    validation_split: float 
    n_channel_in: int 
    pattern: str
    depth: int 
    batch_size: int    

@dataclass
class NeurIPSConfig:

     paths: Paths
     files : Files
     params: Params
     