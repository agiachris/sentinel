"""
Construct a point cloud from multiple ZED cameras.
Starting from the example at github.com/stereolabs/zed-multi-camera.
Note: using globals only because ZED examples suggest this.

@contactrika @yjy0625
"""

import os
import cv2
import time
import numpy as np
from multiprocessing.connection import Client


def adjust_ranges_centered(mask, image_size):
    assert len(mask.shape) == 2
    # Determine the half lengths for both dimensions
    half_h_length = image_size[0] // 2
    half_w_length = image_size[1] // 2

    # Find the indices where the mask is not zero
    ones_indices = np.where(mask != 0)

    # Calculate the center of the 1s region
    center_h = (ones_indices[0].min() + ones_indices[0].max()) // 2
    center_w = (ones_indices[1].min() + ones_indices[1].max()) // 2

    # Adjust the center points to ensure the ranges do not exceed the image dimensions
    # For H dimension
    if center_h - half_h_length // 2 < 0:
        center_h = half_h_length // 2
    elif center_h + half_h_length // 2 > image_size[0]:
        center_h = image_size[0] - half_h_length // 2

    # For W dimension
    if center_w - half_w_length // 2 < 0:
        center_w = half_w_length // 2
    elif center_w + half_w_length // 2 > image_size[1]:
        center_w = image_size[1] - half_w_length // 2

    # Calculate the adjusted ranges
    adjusted_min_h = center_h - half_h_length // 2
    adjusted_max_h = center_h + half_h_length // 2
    adjusted_min_w = center_w - half_w_length // 2
    adjusted_max_w = center_w + half_w_length // 2

    assert adjusted_max_h - adjusted_min_h == half_h_length
    assert adjusted_max_w - adjusted_min_w == half_w_length

    return (
        int(adjusted_min_h),
        int(adjusted_max_h),
        int(adjusted_min_w),
        int(adjusted_max_w),
    )


class ZedCameraClient(object):
    def __init__(self, cam_ids):
        # load zed related libraries only if zed cameras are used
        import pyzed.sl as sl
        import mob_manip_vision.zed_utils as zed_utils

        self.cam_ids = cam_ids
        # self.image_size = (2208, 1242)
        self.image_size = (1920, 1080)  # (2208, 1242) # (960, 540)  # (1280, 720)

        self.cams = dict()
        self.left_images = dict()
        self.right_images = dict()
        self.sl = sl
        for cam_id in cam_ids:
            self.cams[cam_id] = sl.Camera()
            cam_params = sl.InitParameters(
                coordinate_units=sl.UNIT.METER,
                camera_fps=15,
                camera_resolution=sl.RESOLUTION.HD2K,
            )
            cam_params.set_from_serial_number(cam_id)

            err = self.cams[cam_id].open(cam_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ValueError(f"Cannot open camera {cam_ids[0]}: " + repr(err))
            zed_utils.adjust_cam_settings(self.cams[cam_id], False)

            self.left_images[cam_id] = sl.Mat()
            self.right_images[cam_id] = sl.Mat()

        self.runtime_parameters = sl.RuntimeParameters()

    def get_images(self, return_bg=False, verbose=True):
        left_images = []
        right_images = []
        for cam_id in self.cam_ids:
            cam = self.cams[cam_id]
            while not cam.grab(self.runtime_parameters) == self.sl.ERROR_CODE.SUCCESS:
                # wait and try again
                time.sleep(0.1)
                continue

            cam.retrieve_image(self.left_images[cam_id], self.sl.VIEW.LEFT)
            cam.retrieve_image(self.right_images[cam_id], self.sl.VIEW.RIGHT)

            # left_images.append(self.left_images[cam_id].get_data())
            # right_images.append(self.right_images[cam_id].get_data())
            left_images.append(
                cv2.resize(self.left_images[cam_id].get_data(), self.image_size)
            )
            right_images.append(
                cv2.resize(self.right_images[cam_id].get_data(), self.image_size)
            )

        return np.stack(left_images), np.stack(right_images)

    def __del__(self):
        if hasattr(self, "cams"):
            for k, v in self.cams.items():
                v.close()


class TRIZedCameraClient(ZedCameraClient):
    def __init__(
        self,
        cam_ids,
        env_name="dummy",
        prompt_lists=dict(dummy=["dummy"]),
        dummy=False,
        num_points=8192,
        tracker_crop=None,  # (min_h, max_h, min_w, max_w)
    ):
        print(f"[zed.py] Initializing zed camera...")
        if not dummy:
            super().__init__(cam_ids)
        else:
            self.cam_ids = cam_ids

        self.env_name = env_name
        self.num_points = num_points

        # initialize tri PC processor dd
        self.pc_processors = {}
        for cam_id in cam_ids:
            from sentinel.utils.process_real_pc import RealPCProcessor

            MODEL_PATH = os.environ["TRI_MODEL_PATH"]
            CALIB_DIR = os.environ["CALIB_DIR"]
            self.pc_processors[cam_id] = RealPCProcessor(
                MODEL_PATH,
                CALIB_DIR,
                prompt_lists[env_name],
                num_points=num_points,
                tracker_crop=tracker_crop,
            )
        print(f"[zed.py] Initialization finished.")

        self._latest_rgb_time = time.time()  # for multi-process synchronization

    def reset(self):
        for cam_id in self.cam_ids:
            self.pc_processors[cam_id].reset()

    def imgs2pc(self, left_imgs, right_imgs, return_bg=False, verbose=True):
        if verbose:
            st = time.time()
        left_imgs = left_imgs[..., [2, 1, 0]]
        right_imgs = right_imgs[..., [2, 1, 0]]

        raw_pcs, raw_colors = [None] * len(left_imgs), [None] * len(left_imgs)

        cam_pcs, cam_bg_pcs, cam_colors, cam_bg_colors, cam_masks = [], [], [], [], []
        for i in range(len(self.cam_ids)):
            pc_processor = self.pc_processors[self.cam_ids[i]]
            ret_list = pc_processor.color_filter_and_make_pc(
                [left_imgs[i], right_imgs[i]],
                self.cam_ids[i],
                view="side" if self.cam_ids[i] == 21172477 else "top",
                raw_pcs=raw_pcs,
                raw_colors=raw_colors,
                return_bg=return_bg,
                verbose=verbose,
            )
            if return_bg:
                masked, pcs, bg_pcs, colors, bg_colors, seg_image = ret_list
                cam_pcs.append(pcs)
                cam_bg_pcs.append(bg_pcs)
                cam_colors.append(colors)
                cam_bg_colors.append(bg_colors)
                cam_masks.append(seg_image)
            else:
                masked, pcs, colors, seg_image = ret_list
                cam_pcs.append(pcs)
                cam_colors.append(colors)
                cam_masks.append(seg_image)
        if verbose:
            print(
                "\033[1m"
                + f"[zed.py => imgs2pc()] pc processing took {time.time() - st:.3f}s"
                + "\033[0m"
            )
            st = time.time()
        if return_bg:
            return (
                np.concatenate(cam_pcs),
                np.concatenate(cam_bg_pcs),
                np.concatenate(cam_colors),
                np.concatenate(cam_bg_colors),
                np.concatenate([left_imgs, right_imgs]),
                np.stack(cam_masks),
            )
        else:
            return (
                np.concatenate(cam_pcs),
                np.concatenate([left_imgs, right_imgs]),
                np.stack(cam_masks),
            )

    def get_images(self, return_bg=False, verbose=True):
        if verbose:
            st = time.time()

        self._latest_rgb_time = time.time()  # for multi-process synchronization
        left_imgs, right_imgs = super().get_images()
        assert len(left_imgs) == len(self.cam_ids)
        assert len(right_imgs) == len(self.cam_ids)

        if verbose:
            print(
                "\033[1m"
                + f"[zed.py => get_images()] read raw img took {time.time() - st:.3f}s"
                + "\033[0m"
            )
        return self.imgs2pc(left_imgs, right_imgs, return_bg=return_bg, verbose=verbose)


class RemoteTRIZedCameraClient(TRIZedCameraClient):
    def __init__(
        self,
        cam_ids,
        env_name="dummy",
        prompt_lists=dict(dummy="dummy"),
        num_points=8192,
        ip=None,
        port=None,
    ):
        self.remote = ip is not None  # and port is not None
        super().__init__(
            cam_ids,
            env_name=env_name,
            prompt_lists=prompt_lists,
            dummy=True,
            num_points=num_points,
        )
        if self.remote:
            self.conns = [
                Client((ip, cam % 10000), authkey=b"secret") for cam in cam_ids
            ]

    def __del__(self):
        if self.remote and hasattr(self, "conns"):
            for conn in self.conns:
                conn.close()

    def decompress_array(self, arr, shape, arr_dtype=np.uint8):
        image_array = np.frombuffer(arr, dtype=arr_dtype).reshape(shape)
        return image_array

    def reset(self):
        super().reset()

    def get_images(self, return_bg=False, verbose=True):
        if self.remote:
            if verbose:
                st = time.time()
            for conn in self.conns:
                conn.send(None)
            left_imgs, right_imgs = [], []
            for conn in self.conns:
                [encoded_image, sender_time] = conn.recv()
                print(
                    "\033[1m"
                    + f"[zed.py => get_images()] just transfer took {time.time() - st:.3f}s; diff {time.time() - sender_time:.3f}s"
                    + "\033[0m"
                )
                image = np.frombuffer(encoded_image, dtype=np.uint8).reshape(
                    1920, 540, 3
                )
                # image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
                left_imgs.append(image[: len(image) // 2])
                right_imgs.append(image[len(image) // 2 :])
            left_imgs = np.array(left_imgs)
            right_imgs = np.array(right_imgs)
            if verbose:
                print(
                    "\033[1m"
                    + f"[zed.py => get_images()] image transfer took {time.time() - st:.3f}s"
                    + "\033[0m"
                )
            ret_list = self.imgs2pc(
                left_imgs, right_imgs, return_bg=return_bg, verbose=True
            )
            if verbose:
                print(
                    "\033[1m"
                    + f"[zed.py => get_images() / summary] entire image processing pipeline took {time.time() - st:.3f}s"
                    + "\033[0m"
                )
            return tuple(ret_list)
        else:
            return super().get_images()


if __name__ == "__main__":
    zed = TRIZedCameraClient(
        [26279249], env_name="close", prompt_lists=dict(close=["brown box"])
    )

    for i in range(10):
        start_time = time.time()
        pcs, images = zed.get_images()
        print(
            f"Got point cloud with shape {pcs.shape} in {time.time() - start_time:.3f}s"
        )
        im = images.reshape(2 * 1242, 2208, 3).copy()
        cv2.imwrite("im.jpg", im)
