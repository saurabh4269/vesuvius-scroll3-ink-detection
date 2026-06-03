#!/usr/bin/env python
"""Prepare augmented ESRF training dataset for improved generalization."""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import albumentations as A


def load_patches_from_directory(frag_dir, output_subdir='patches'):
    """Load all patches from a fragment directory."""
    frag_path = Path(frag_dir)
    patches_dir = frag_path / output_subdir

    if not patches_dir.exists():
        print(f"[AUGMENT] No patches found in {patches_dir}")
        return [], []

    images = []
    labels = []

    # Load all patches
    for img_file in sorted(patches_dir.glob('*.png')):
        try:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((img, img_file.name))
        except Exception as e:
            print(f"[AUGMENT] Error loading {img_file}: {e}")

    print(f"[AUGMENT] Loaded {len(images)} image patches from {frag_dir}")
    return images


def create_augmentation_pipeline():
    """Create albumentations augmentation pipeline."""
    return A.Compose([
        # Rotation
        A.Rotate(limit=15, p=0.7),

        # Contrast and brightness
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.2, 0.2), p=0.6),
        A.GaussianBlur(blur_limit=3, p=0.3),

        # Noise
        A.GaussNoise(p=0.4, var_limit=(10.0, 20.0)),

        # Elastic deformation (simulates papyrus warping)
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3),

        # Flip occasionally
        A.Flip(p=0.3),
    ], keypoint_params=A.KeypointParams(format='xy'))


def augment_dataset(image_files, output_dir, num_augmentations=3, seed=42):
    """
    Augment dataset with specified number of variations per image.

    Args:
        image_files: List of (image, filename) tuples
        output_dir: Output directory for augmented images
        num_augmentations: Number of augmentation variations per image
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmenter = create_augmentation_pipeline()
    np.random.seed(seed)

    augmented_count = 0

    for img, filename in image_files:
        # Save original
        orig_path = output_dir / f"original_{filename}"
        cv2.imwrite(str(orig_path), img)
        augmented_count += 1

        # Create augmentation variations
        for aug_idx in range(num_augmentations):
            try:
                augmented = augmenter(image=img)['image']
                aug_filename = f"aug{aug_idx}_{filename}"
                aug_path = output_dir / aug_filename
                cv2.imwrite(str(aug_path), augmented)
                augmented_count += 1
            except Exception as e:
                print(f"[AUGMENT] Error augmenting {filename} (variation {aug_idx}): {e}")

    print(f"[AUGMENT] ✓ Created {augmented_count} total images ({len(image_files)} originals + {num_augmentations} variations each)")
    return augmented_count


def analyze_augmentation_impact(original_dir, augmented_dir):
    """Analyze the impact of augmentation on dataset statistics."""
    print("\n[AUGMENT] Dataset Analysis")
    print("=" * 60)

    original_count = len(list(Path(original_dir).glob('*.png')))
    augmented_count = len(list(Path(augmented_dir).glob('*.png')))

    print(f"Original dataset: {original_count} images")
    print(f"Augmented dataset: {augmented_count} images")
    print(f"Effective size increase: {augmented_count / original_count:.1f}x")

    # Load sample augmented image to verify
    sample_files = list(Path(augmented_dir).glob('*.png'))[:5]
    if sample_files:
        sample = cv2.imread(str(sample_files[0]), cv2.IMREAD_GRAYSCALE)
        print(f"\nSample augmented image shape: {sample.shape}")
        print(f"Sample value range: [{sample.min()}, {sample.max()}]")


def main(input_dir, output_dir, num_augmentations=3, fragments=None):
    """Main augmentation pipeline."""
    print("[AUGMENT] Starting ESRF dataset augmentation...")
    print(f"[AUGMENT] Input directory: {input_dir}")
    print(f"[AUGMENT] Output directory: {output_dir}")
    print(f"[AUGMENT] Augmentations per image: {num_augmentations}")
    print()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If fragments specified, process each separately
    if fragments:
        total_augmented = 0
        for frag in fragments:
            frag_dir = input_dir / frag / 'patches'
            if frag_dir.exists():
                print(f"\n[AUGMENT] Processing {frag}...")
                images = load_patches_from_directory(frag_dir.parent)

                frag_output = output_dir / frag
                augmented = augment_dataset(images, frag_output, num_augmentations)
                total_augmented += augmented

        print(f"\n[AUGMENT] ✓✓✓ Total augmented: {total_augmented} images")
    else:
        # Process all patches in input directory
        print("[AUGMENT] Processing all patches...")
        # This would be for a single flat structure
        images = load_patches_from_directory(str(input_dir))
        augmented = augment_dataset(images, output_dir, num_augmentations)

    print("\n[AUGMENT] ✓ Augmentation complete!")
    print(f"[AUGMENT] Use {output_dir} for training with augmented data")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment ESRF training dataset')
    parser.add_argument('input_dir', type=str, help='Input directory with original patches')
    parser.add_argument('--output-dir', type=str, default='~/scroll_prize/data/esrf/augmented',
                        help='Output directory for augmented patches')
    parser.add_argument('--num-augmentations', type=int, default=3,
                        help='Number of augmentation variations per image')
    parser.add_argument('--fragments', nargs='+', default=['500P2', '343P'],
                        help='Fragment names to augment')

    args = parser.parse_args()
    args.input_dir = Path(args.input_dir).expanduser()
    args.output_dir = Path(args.output_dir).expanduser()

    sys.exit(main(args.input_dir, args.output_dir, args.num_augmentations, args.fragments))
