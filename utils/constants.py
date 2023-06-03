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

# Model constants
# Model only upscales to a factor of 3
# do not change unless you also change the generator
UPSCALE_FACTOR = 3
