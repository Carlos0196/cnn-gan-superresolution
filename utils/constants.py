# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

# Training constants
HR_IMG_SIZE = 300
LR_IMG_SIZE = 100
BATCH_SIZE = 8
SEED = 42

# Visualization constants
ZOOM_FOR_LR = [83, 108, 33, 58]
ZOOM_FOR_LR_SQUARE = [50, 66, 33, 50]
ZOOM_FOR_HR = [250, 325, 100, 175]
ZOOM_FOR_HR_SQUARE = [150, 200, 100, 150]
SAMPLE_IMAGES_VAL = [
    'val/108070.jpg',
    'val/299086.jpg',
    'val/210088.jpg',
    'val/41069.jpg',
    'val/86000.jpg'
]

# Dataframe constants
DF_COLUMNS = [
    'BIC_MSE',
    'GEN_MSE',
    'BIC_SSIM',
    'GEN_SSIM',
    'BIC_PSNR',
    'GEN_PSNR'
]

# Model constants
# Model only upscales to a factor of 3
# do not change unless you also change the generator
UPSCALE_FACTOR = 3
