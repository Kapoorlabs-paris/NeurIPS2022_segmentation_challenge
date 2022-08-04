
from SmartPatch2D import SmartPatch2D
from config_data import NeurIPSDataConfig

import hydra
from hydra.core.config_store import ConfigStore


configstore = ConfigStore.instance()
configstore.store(name = 'neurips_data_config', node = NeurIPSDataConfig)

@hydra.main(config_path="conf", config_name='config_data')
def main(config : NeurIPSDataConfig):

            base_dir = config.paths.base_dir
            raw_dir = config.paths.raw_dir
            raw_save_dir = config.paths.raw_save_dir
            real_mask_dir = config.paths.real_mask_dir 
            binary_mask_dir = config.paths.binary_mask_dir
            binary_erode_mask_dir = config.paths.binary_erode_mask_dir
            #Network training parameters
            patch_x = config.params.patch_x
            patch_y = config.params.patch_y
            pattern = config.params.pattern
            num_channels = config.params.num_channels
            lower_ratio_fore_to_back = config.params.lower_ratio_fore_to_back
            upper_ratio_fore_to_back = config.params.upper_ratio_fore_to_back
            SmartPatch2D(base_dir, (patch_y, patch_x), num_channels,  raw_dir = raw_dir, raw_save_dir = raw_save_dir,
     real_mask_dir = real_mask_dir, binary_mask_dir = binary_mask_dir, binary_erode_mask_dir = binary_erode_mask_dir,  pattern = pattern,
     lower_ratio_fore_to_back = lower_ratio_fore_to_back,upper_ratio_fore_to_back= upper_ratio_fore_to_back )
           


if __name__ == '__main__':
    main()             