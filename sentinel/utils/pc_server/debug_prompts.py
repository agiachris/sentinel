import numpy as np
import os
import cv2
import json
import time
import click
import itertools
from glob import glob
from sentinel.utils.zed import TRIZedCameraClient


loop_time = 0.1


@click.command()
@click.option(
    "--image_dir",
    type=str,
)
@click.option(
    "--camera_id",
    type=int,
    #   default=27622607,
    default=21582473,
)
@click.option("--env_name", type=str, default="dummy")
@click.option(
    "--prompt",
    type=str,
    default="wood and white table, woven bowl near the table",
)
@click.option(
    "--calib_dir",
    type=str,
)
@click.option("--flip", is_flag=False)
@click.option("--crop", type=str, default=None)
def main(image_dir, camera_id, env_name, prompt, calib_dir, flip, crop):
    os.environ["CALIB_DIR"] = calib_dir
    camera_ids = [camera_id]
    if crop is not None:
        crop = [int(s) for s in crop.split(",")]
    camera = TRIZedCameraClient(
        camera_ids,
        env_name=env_name,
        prompt_lists={env_name: prompt.split(",")},
        num_points=2048,
        dummy=True,
        tracker_crop=crop,
    )
    camera.reset()

    print("\033[1m" + "Loading finished." + "\033[0m")
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    left_img_paths = list(sorted(glob(os.path.join(image_dir, "demo*_t*_left.jpg"))))
    for left_img_path in itertools.cycle(left_img_paths):
        start_time = time.time()
        right_img_path = left_img_path.replace("left", "right")
        left_imgs = cv2.imread(left_img_path)[None]
        right_imgs = cv2.imread(right_img_path)[None]

        point_cloud, image, mask = camera.imgs2pc(left_imgs, right_imgs)
        elapsed_time = time.time() - start_time

        process_start_time = time.time()

        # Subsample point cloud
        subsampled_pc = subsample_point_cloud(point_cloud)

        # Save point cloud data
        with open(f"{UPLOAD_FOLDER}/point_cloud.json", "w") as pc_file:
            json.dump(subsampled_pc.tolist(), pc_file)

        # Save image
        if flip:
            image[0] = image[0][::-1, ::-1]
        cv2.imwrite(f"{UPLOAD_FOLDER}/image.jpg", image[0][..., ::-1])

        # Save mask
        if flip:
            mask[0] = mask[0][::-1, ::-1]
        cv2.imwrite(f"{UPLOAD_FOLDER}/mask.png", mask[0][..., None].astype(int) * 255)

        # Save elapsed time
        with open(f"{UPLOAD_FOLDER}/elapsed_time.txt", "w") as et_file:
            et_file.write(str(elapsed_time))

        process_elapsed_time = time.time() - process_start_time

        with open(f"{UPLOAD_FOLDER}/process_elapsed_time.txt", "w") as et_file:
            et_file.write(str(process_elapsed_time))

        time.sleep(max(0.0, loop_time - (time.time() - start_time)))


# Function to subsample point cloud
def subsample_point_cloud(point_cloud, num_points=512):
    if point_cloud is None or len(point_cloud) == 0:
        return point_cloud
    return point_cloud[np.random.choice(point_cloud.shape[0], num_points, replace=True)]


if __name__ == "__main__":
    main()
