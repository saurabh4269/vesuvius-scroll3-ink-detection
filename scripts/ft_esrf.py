"""
Training config for MiniUNETR on ESRF fragments 500P2 + 343P.
Place in: ~/scroll_prize/vesuvius_first_title_prize/configs/ft_esrf.py
"""
import os
import albumentations as A

# Relative paths from ~/scroll_prize/vesuvius_first_title_prize/ (the CWD when scripts run)
data_root_dir = '../data/esrf'
dataset_target_dir = '../data/esrf/datasets/ft_esrf_500P2_343P'

# Architecture
model_type = 'b3'
architecture = 'unetr-sf'
segformer_from_pretrained = f'nvidia/mit-{model_type}'
model_name = 'unetr-sf-b3'
mini_unetr = True
unetr_out_channels = 32
in_chans = 16
patch_size = 128
label_size = 32
stride = 128     # 64 would generate ~22 GB of patches; 128 gives ~5 GB (fits home dir quota)

# Trainer
epochs = -1           # train until convergence (use early stopping / manual stop)
node = True
num_workers = 4      # 4 workers; 0 was deadlocking with num_workers=8 but 0 was too slow; use persistent_workers via Lightning
seed = 7340043
val_interval = 1
gradient_clip_val = 1

# Data — scroll0 is our fake scroll ID for ESRF data
scroll_id = 0
contrasted = False    # ESRF layers already have good contrast at 2.2 µm — skip LUT
val_frac = 0.1        # 10% val (more than 5% since we have fewer fragments)
dataset_fraction = 1
take_full_dataset = False
ink_ratio = 5         # both frags have ~4% ink → 5:1 non-ink:ink keeps ~17% ink patches (same as S5 config)
no_ink_sample_percentage = 0.75
train_batch_size = 32
val_batch_size = 64

# Both ESRF fragments
fragment_ids = [
    "500P2",
    "343P",
]
validation_fragments = []

# Optimizer
weight_decay = 0.001
label_smoothing = 0.1

# Learning rate schedule
lr = 2e-04
epsilon = 0.001
cos_eta_min = 1e-06
cos_max_epochs = 50

# Augmentations (same as First Title winner)
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
