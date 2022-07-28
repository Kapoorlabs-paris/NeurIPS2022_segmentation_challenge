
from vollseg import SmartSeeds2D


base_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/'
npz_filename = 'Kapoorlabs_NeuroIPS_p128'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_Models/'
model_name = 'Kapoorlabs_NeuroIPS_p128_d3_f32_r64'

raw_dir = 'Raw/'
real_mask_dir = 'real_mask/' 
binary_mask_dir = 'binary_mask/'
binary_erode_mask_dir = 'binary_erode_mask/'



#Network training parameters
depth = 3
epochs = 200
learning_rate = 1.0E-4
batch_size = 10
patch_x = 128
patch_y = 128
kern_size = 3
n_patches_per_image = 16
n_rays = 64
startfilter = 32
validation_split = 0.01
n_channel_in = 1
use_gpu_opencl = True
load_data_sequence = False
generate_npz = True
train_unet = True
train_star = True
train_seed_unet = True

pattern = '.tiff'

SmartSeeds2D(base_dir = base_dir, 
             npz_filename = npz_filename, 
             model_name = model_name, 
             model_dir = model_dir,
             raw_dir = raw_dir,
             real_mask_dir = real_mask_dir,
             binary_mask_dir = binary_mask_dir,
             binary_erode_mask_dir = binary_erode_mask_dir,
             n_channel_in = n_channel_in,
             load_data_sequence = load_data_sequence, 
             validation_split = validation_split, 
             n_patches_per_image = n_patches_per_image, 
             generate_npz = generate_npz, 
             train_unet = train_unet, 
             train_star = train_star, 
             train_seed_unet = train_seed_unet,
             patch_x= patch_x, 
             patch_y= patch_y, 
             use_gpu = use_gpu_opencl,  
             batch_size = batch_size, 
             depth = depth, 
             pattern = pattern,
             kern_size = kern_size, 
             startfilter = startfilter, 
             n_rays = n_rays, 
             epochs = epochs, 
             learning_rate = learning_rate)