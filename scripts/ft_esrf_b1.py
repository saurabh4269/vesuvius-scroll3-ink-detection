"""
Training config for MiniUNETR with Segformer-B1 backbone on ESRF fragments.
Segformer B1: ~14M params vs B3's ~47M → better generalization on 3,276 patches.
"""
import albumentations as A

data_root_dir = '../data/esrf'
dataset_target_dir = '../data/esrf/datasets/ft_esrf_500P2_343P'

# B1 architecture — smaller, better suited for limited data
model_type = 'b1'
architecture = 'unetr-sf'
segformer_from_pretrained = 'nvidia/mit-b1'
model_name = 'unetr-sf-b1'
mini_unetr = True
unetr_out_channels = 32
in_chans = 16
patch_size = 128
label_size = 32
stride = 128

epochs = -1
node = True
num_workers = 0
seed = 7340043
val_interval = 999
gradient_clip_val = 1

scroll_id = 0
contrasted = False
val_frac = 0.01
dataset_fraction = 1
take_full_dataset = False
ink_ratio = 5
no_ink_sample_percentage = 0.75
train_batch_size = 32
val_batch_size = 64

fragment_ids = [
    "500P2",
    "343P",
]
validation_fragments = []

weight_decay = 0.001
label_smoothing = 0.1

lr = 2e-04
epsilon = 0.001
cos_eta_min = 1e-06
cos_max_epochs = 50

train_aug = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=360, p=0.5),
    A.Perspective(scale=(0.03, 0.03), p=0.1),
    A.GridDistortion(p=0.1),
    A.Blur(blur_limit=3, p=0.1),
    A.GaussNoise(p=0.1),
    A.RandomResizedCrop(size=(patch_size, patch_size), scale=(0.5, 1.0), ratio=(0.75, 1.333), p=0.15),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=360, p=0.1),
    A.RandomGamma(p=0.15, gamma_limit=(30, 80)),
    A.RandomBrightnessContrast(p=0.15, brightness_limit=(-0.2, 0.4), contrast_limit=(-0.2, 0.2)),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
]
val_aug = [
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
]
