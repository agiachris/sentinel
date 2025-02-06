import numpy as np
import redis


HOSTS = {
    "bot1": "172.24.69.149",
    "bot2": "172.24.69.150",
    "bot3": "172.24.69.151",
    "bot4": "172.24.69.155",
}
PASSWORD = "bohg"


class MobileBase(object):
    def __init__(self, name="bot1", password=PASSWORD):
        self.name = name
        self.host = HOSTS[name]
        self.base = redis.Redis(host=self.host, password=password)
        # self.set_max_velocity(0.5, 0.5, 3.14)
        self.set_max_velocity(0.1, 0.1, 3.14)

        self.set_max_acceleration(0.5, 0.5, 2.36)
        self._stopped = True
        self._ref_ang = None

    def get_pose(self):
        s = self.base.get(f"mmp::{self.name}::veh::sensor::x")
        x = np.array([float(ss) for ss in s.decode("utf-8").split(" ")])
        return x

    def get_velocity(self):
        s = self.base.get(f"mmp::{self.name}::veh::sensor::dx")
        x = np.array([float(ss) for ss in s.decode("utf-8").split(" ")])
        return x

    def close_to_pose(self, pose, threshold=np.array([0.01, 0.01, 0.05])):
        is_close = np.all(np.abs(self.get_pose() - pose) <= threshold)
        return is_close

    def set_ref_ang(self):
        self._ref_ang = self.get_pose()[-1]
        print(f"[mobile base] setting ref ang to {self._ref_ang}")

    def goto_pose(
        self,
        pose,
    ):
        assert type(pose) == np.ndarray
        assert len(pose) == 3
        if self._ref_ang is not None:
            ang = pose[2]
            ang = self._ref_ang + np.mod(ang - self._ref_ang + np.pi, np.pi * 2) - np.pi
            pose[2] = ang
        self._last_ang = pose[2]
        pose_str = f"{pose[0]} {pose[1]} {pose[2]}"
        self.base.set(f"mmp::{self.name}::veh::control::x", pose_str)
        ok = True
        return ok

    def emergency_shutdown(self):
        self.base.set(f"mmp::emergency_shutdown", 1)

    def stop(self):
        self.base.set(f"mmp::{self.name}::veh::stop", 1)
        print(
            f"mmp::{self.name}::veh::stop:",
            self.base.get(f"mmp::{self.name}::veh::stop"),
        )
        self._stopped = True

    def go(self):
        self.base.set(f"mmp::{self.name}::veh::stop", 0)
        self._stopped = False

    def set_max_velocity(self, max_vel_x, max_vel_y, max_vel_theta):
        self.base.set(
            f"mmp::{self.name}::veh::control::max_vel",
            f"{max_vel_x} {max_vel_y} {max_vel_theta}",
        )

    def set_max_acceleration(self, max_accel_x, max_accel_y, max_accel_theta):
        self.base.set(
            f"mmp::{self.name}::veh::control::max_accel",
            f"{max_accel_x} {max_accel_y} {max_accel_theta}",
        )


def main():
    mobile_base = MobileBase(name="bot3")
    print(mobile_base.get_pose())
    target_pose = np.array([0.5, 0.5, 3.14])
    print(f"Moving base to {target_pose}")
    mobile_base.goto_pose(target_pose)
    print("done")


if __name__ == "__main__":
    main()
