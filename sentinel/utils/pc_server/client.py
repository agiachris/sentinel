import numpy as np
import os
import cv2
import json
import time
from sentinel.utils.zed import TRIZedCameraClient
import faulthandler

faulthandler.enable()

# Initialize camera
# camera_ids = [28221883] # [21172477], [21582473], [28329710] [27622607] [27667894]
camera_ids = [21582473]
camera = TRIZedCameraClient(
    camera_ids,
    # env_name="fold", # "laundry_door", # "push_chair",
    env_name="push_chair",  # "laundry_door", # "push_chair",
    prompt_lists=dict(
        make_bed=["gray towel"],
        fold=["grey blanket"],
        close_luggage=["open luggage"],
        laundry_door=["left laundry machine"],
        laundry_load=["blue cloth", "black open laundry machine"],
        #   load_shoes=["beige slippers", "shoe rack"],
        load_shoes=["beige shoes pair", "silver rack"],
        packing=[
            "open luggage",
            "bag of white shirt",
        ],  # ["opened luggage", "bag with white shirt", "items in luggage"],
        lift=["woven basket", "wood and white table"],
        push_chair=["black chair", "desk"],
    ),  # close_luggage=["desk", "black chair"]),
    #   laundry_door=["left laundry machine"],
    num_points=2048,
)
"""
camera = RemoteTRIZedCameraClient(
    camera_ids,
    env_name="fold",
    prompt_lists=dict(fold=["colorful cloth"]),
    num_points=2048,
    ip="iprl-orin1",
)
"""
camera.reset()

print("\033[1m" + "Loading finished." + "\033[0m")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")


# Function to subsample point cloud
def subsample_point_cloud(point_cloud, num_points=512):
    if point_cloud is None or len(point_cloud) == 0:
        return point_cloud
    return point_cloud[np.random.choice(point_cloud.shape[0], num_points, replace=True)]


loop_time = 0.1

while True:
    start_time = time.time()
    point_cloud, image, mask = camera.get_images()
    elapsed_time = time.time() - start_time

    process_start_time = time.time()

    # Subsample point cloud
    subsampled_pc = subsample_point_cloud(point_cloud)

    # Save point cloud data
    with open(f"{UPLOAD_FOLDER}/point_cloud.json", "w") as pc_file:
        json.dump(subsampled_pc.tolist(), pc_file)

    # Save image
    cv2.imwrite(f"{UPLOAD_FOLDER}/image.jpg", image[0][..., ::-1])

    # Save mask
    cv2.imwrite(f"{UPLOAD_FOLDER}/mask.png", mask[0][..., None].astype(int) * 255)

    # Save elapsed time
    with open(f"{UPLOAD_FOLDER}/elapsed_time.txt", "w") as et_file:
        et_file.write(str(elapsed_time))

    process_elapsed_time = time.time() - process_start_time

    with open(f"{UPLOAD_FOLDER}/process_elapsed_time.txt", "w") as et_file:
        et_file.write(str(process_elapsed_time))

    time.sleep(max(0.0, loop_time - (time.time() - start_time)))
