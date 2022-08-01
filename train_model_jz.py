
from vollseg import SmartSeeds2D
import argparse




base_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_Models/'

parser = argparse.ArgumentParser(description='Multiple trainings.')
parser.add_argument('--generate_npz', help='generate_npz', action='store_true')
parser.add_argument('--npz_filename', type = str, default = 'Kapoorlabs_NeuroIPS_p128', help='npz_filename')
parser.add_argument('--model_name', type = str, default = 'Kapoorlabs_NeuroIPS_p128_d3_f32_r64', help='model_name')
parser.add_argument('--epoch', help='e', type= int, default=200)
parser.add_argument('--depth', help='d', type= int, default=3)
parser.add_argument('--batch_size', help='b', type= int, default=10)
parser.add_argument('--learning_rate', help='lr', type = float, default = 1.0E-4)
parser.add_argument('--patch_x', help='px', type= int, default=128)
parser.add_argument('--patch_y', help='py', type= int, default=128)
parser.add_argument('--kern_size', help='ks', type= int, default=3)
parser.add_argument('--n_patches_per_image', help='npi', type= int, default=32)
parser.add_argument('--n_rays', help='nr', type= int, default=64)
parser.add_argument('--startfilter', help='sf', type= int, default=32)
parser.add_argument('--validation_split', help='vs', type = float, default = 0.0001)
parser.add_argument('--n_channel_in', help='nin', type= int, default=3)
parser.add_argument('--pattern', type = str, default = '.tiff', help='.tiff')

args = parser.parse_args()

npz_filename = args.npz_filename

model_name = args.model_name

raw_dir = 'raw/'
real_mask_dir = 'real_mask/' 
binary_mask_dir = 'binary_mask/'
binary_erode_mask_dir = 'binary_erode_mask/'


print('Flags', args)
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
