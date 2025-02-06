import os
import cv2
import click
import numpy as np
from glob import glob

from sentinel.utils.media import save_video


@click.command()
@click.option("--input_dir", type=str)
def main(input_dir):
    filenames = list(glob(os.path.join(input_dir, "demo*_t0001_left.jpg")))
    images = []
    for filename in filenames:
        img = cv2.imread(filename)[..., ::-1]
        images.append(img)
    save_path = os.path.join(input_dir, "initial_poses.gif")
    save_video(np.array(images), save_path, fps=60)
    print(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()
