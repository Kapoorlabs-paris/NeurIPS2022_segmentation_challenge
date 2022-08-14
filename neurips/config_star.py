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
  model_name : str

@dataclass
class Params:

    epoch: int
    learning_rate: float
    patch_x: int 
    patch_y: int 
    kern_size: int 
    n_rays: int 
    startfilter: int 
    validation_split: float 
    n_channel_in: int 
    pattern: str
    depth: int 
    batch_size: int    
    use_gpu_opencl: bool 
    load_data_sequence: bool
    RGB: bool
    def_shape_x : int
    def_shape_y: int
    def_shape_c: int

@dataclass
class NeurIPSStarConfig:

     paths: Paths
     files : Files
     params: Params
     