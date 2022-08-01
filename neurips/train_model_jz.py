
from vollseg import SmartSeeds2D
import argparse
import hydra



base_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_Models/'

parser = argparse.ArgumentParser(description='Multiple trainings.')
parser.add_argument( '-g','--generate_npz', help='generate npz files if not already there', action='store_true')
parser.add_argument( '-npz','--npz_filename', type = str, default = 'Kapoorlabs_NeuroIPS_p128', help='name of the npz file to put')
parser.add_argument( '-m', '--model_name', type = str, default = 'Kapoorlabs_NeuroIPS_p128_d3_f32_r64', help='name of the model to put')
parser.add_argument( '-e', '--epoch', help='number of epochs to train the models for', type= int, default=200)
parser.add_argument( '-d', '--depth', help='depth of the network', type= int, default=3)
parser.add_argument( '-b', '--batch_size', help='batch_size', type= int, default=10)
parser.add_argument( '-lr', '--learning_rate', help='learning_rate', type = float, default = 1.0E-4)
parser.add_argument( '-px', '--patch_x', help='patch size in x for training', type= int, default=128)
parser.add_argument( '-py', '--patch_y', help='patch size in y for training', type= int, default=128)
parser.add_argument( '-k', '--kern_size', help='kernel_size', type= int, default=3)
parser.add_argument( '-npi', '--n_patches_per_image', help='number of patches gnerated per image for training', type= int, default=32)
parser.add_argument( '-nr', '--n_rays', help='number of rays for stardist model', type= int, default=64)
parser.add_argument( '-s', '--startfilter', help='start number of convolutional filters for the model', type= int, default=32)
parser.add_argument( '-v', '--validation_split', help='split fraction between the traiing and validation dataset', type = float, default = 0.0001)
parser.add_argument( '-nin', '--n_channel_in', help='number of input channels', type= int, default=3)
parser.add_argument( '-p', '--pattern', type = str, default = '.tiff', help='The input file extension')

args = parser.parse_args()

npz_filename = args.npz_filename

model_name = args.model_name

raw_dir = 'raw/'
real_mask_dir = 'real_mask/' 
binary_mask_dir = 'binary_mask/'
binary_erode_mask_dir = 'binary_erode_mask/'



#Network training parameters
depth = args.depth
epochs = args.epoch
learning_rate = args.learning_rate
batch_size = args.batch_size
patch_x = args.patch_x
patch_y = args.patch_y
kern_size = args.kern_size
n_patches_per_image = args.n_patches_per_image
n_rays = args.n_rays
startfilter = args.startfilter
validation_split = args.validation_split
n_channel_in = args.n_channel_in
pattern = args.pattern


use_gpu_opencl = True
load_data_sequence = False
generate_npz = False
train_unet = True
train_star = True
train_seed_unet = True


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
             RGB = True,
             n_rays = n_rays, 
             epochs = epochs, 
             learning_rate = learning_rate)
