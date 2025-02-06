import sys
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from multiprocessing.connection import Client


def get_camera_by_name(camera_config):
    camera_name = camera_config.camera_name
    if camera_name == "dummy":
        return DummyCamera()
    elif camera_name == "logitech":
        return LogitechCamera(port=6005)
    elif camera_name == "logitech_v2":
        return LogitechV2Camera(addresses=camera_config.addresses)
    elif camera_name == "realsense":
        pass
    else:
        raise ValueError(f"Camera {camera_name} not found.")


class Camera(object):
    def __init__(self):
        pass

    def get_image(self):
        raise NotImplementedError()

    def get_base_poses(self):
        raise NotImplementedError()


class DummyCamera(Camera):
    def __init__(self, port=None):
        pass

    def get_image(self):
        return np.zeros((720, 640, 3))

    def get_base_poses(self):
        return None


class LogitechCamera(Camera):
    def __init__(self, port=7010):
        address = "localhost"
        password = b"secret password"
        print(f"[logitech cam] establishing connection...", end=" ")
        sys.stdout.flush()
        self.conn = Client((address, port), authkey=password)
        print("done.")

    def get_image(self):
        num_retries = 0
        while True:
            num_retries += 1
            if num_retries > 1:
                print(f"Retrying get camera image (attempt {num_retries})")
            self.conn.send({"type": "image"})
            data = self.conn.recv()
            if "image" in data:
                images = [data["image"]]
                break

        return np.concatenate(images, axis=0)[..., ::-1]

    def get_base_poses(self, verbose=False):
        if verbose:
            print(f"[logitech cam] getting base pose...", end=" ")
        self.conn.send({"type": "pose"})

        num_retries = 0
        while True:
            num_retries += 1
            if num_retries > 1:
                print(f"Retrying get base poses (attempt {num_retries})")
            data = self.conn.recv()
            if "poses" in data:
                poses = data["poses"]
                poses = {
                    k: np.array(
                        [-v[1], v[0], np.mod(np.pi * 3 / 2 + v[2], np.pi * 2) - np.pi]
                    )
                    for k, v in poses.items()
                }
                break

        if verbose:
            print("done.")
        return poses


class LogitechV2Camera(Camera):
    def __init__(self, addresses=[]):
        password = b"secret"
        print(f"[logitech cam] establishing connection...", end=" ")
        sys.stdout.flush()
        self.conns = []
        self.ids = []
        for address in addresses:
            self.ids.append(int(address.ip[-1]) - 1)
            self.conns.append(Client((address.ip, address.port), authkey=password))

    def get_image(self):
        raise NotImplementedError()

    def get_base_poses(self, verbose=False):
        if verbose:
            print(f"[logitech cam] getting base pose...", end=" ")
        base_poses = dict()
        for robot_id, conn in zip(self.ids, self.conns):
            conn.send(None)
            base_pose = conn.recv()
            base_poses[robot_id] = base_pose
        return base_poses


def visualize_robot_poses(poses, enable_vis=False):
    if not enable_vis:
        return

    plt.ion()  # Turn on interactive mode
    plt.clf()  # Clear the figure
    for i, pose in enumerate(poses):
        x, y, theta = pose
        # Calculate the orientation of the robot for the arrow
        dx = 0.25 * np.cos(theta)
        dy = 0.25 * np.sin(theta)

        plt.plot([x - 0.25, x + 0.25], [y, y], "k-", lw=2)  # Robot body as a line
        plt.arrow(
            x, y, dx, dy, head_width=0.1, head_length=0.15, fc="k", ec="k"
        )  # Heading

        plt.text(x, y, str(i), color="red", fontsize=12, ha="center")  # Index label

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    enable_vis = True  # Set to False to disable visualization
    addresses = OmegaConf.create(
        [{"ip": "iprl-bot1", "port": 6100}, {"ip": "iprl-bot3", "port": 6100}]
    )
    cam = LogitechV2Camera(addresses=addresses)
    plt.figure(figsize=(8, 8))  # Prepare the plot
    while True:
        import time

        st = time.time()
        poses = cam.get_base_poses()  # Assuming this returns a list of [x, y, theta]
        print(poses)
        print(f"Time: {time.time() - st:.3f}s")

        if enable_vis:
            visualize_robot_poses(poses, enable_vis)

        time.sleep(0.01)
