import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def generate_lane_image(width=512, height=256):
    """Generate a simple synthetic lane image and mask."""

    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Road background
    img[:] = (50, 50, 50)

    # Lane curve parameters
    x_center = width // 2
    curve_strength = np.random.randint(-40, 40)

    for y in range(height):
        x_left = int(x_center - 80 + (y / height) * curve_strength)
        x_right = int(x_center + 80 + (y / height) * curve_strength)

        # Draw lanes (white)
        cv2.line(img, (x_left, y), (x_left, y), (255, 255, 255), 2)
        cv2.line(img, (x_right, y), (x_right, y), (255, 255, 255), 2)

        # Draw mask (1 = lane)
        cv2.line(mask, (x_left, y), (x_left, y), 1, 2)
        cv2.line(mask, (x_right, y), (x_right, y), 1, 2)

    return img, mask


def generate_dataset(out_dir, n=200):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    print(f"Generating {n} lane images...")

    for i in tqdm(range(n)):
        img, mask = generate_lane_image()

        cv2.imwrite(os.path.join(out_dir, "images", f"lane_{i:04}.png"), img)
        cv2.imwrite(os.path.join(out_dir, "masks", f"lane_{i:04}.png"), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/lanes_synthetic")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    generate_dataset(args.out, args.n)
