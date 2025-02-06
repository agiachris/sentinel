# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import numpy as np
import sys
import time
from sentinel.utils.mocap_client.natnet_client import IN_DOMESTIC_SUITE, NatNetClient


from multiprocessing import Lock


class MocapAgent(object):
    def __init__(
        self,
        ip=(
            "172.24.69.162" if IN_DOMESTIC_SUITE else "172.24.68.19"
        ),  # NOTE: Domestic Suite or Dog Room
        use_multicast=True,
        focus_bot_name="bot2",
        shared_memory=None,
        shared_memory_shm=None,
    ):

        self._save_to_shm = False

        if shared_memory is not None:
            self._save_to_shm = True
            self._focus_bot_name = focus_bot_name
            self._shared_memory = shared_memory
            self._shared_memory_shm = shared_memory_shm

            self._shared_memory_lock = Lock()

        self._rigid_body_data = {}

        streaming_client = NatNetClient()
        streaming_client.set_client_address("127.0.0.1")
        streaming_client.set_server_address(ip)
        streaming_client.set_use_multicast(use_multicast)

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        streaming_client.new_frame_listener = self.receive_new_frame

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = streaming_client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting 1")

        time.sleep(1)
        if streaming_client.connected() is False:
            print(
                "ERROR: Could not connect properly.  Check that Motive streaming is on."
            )
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting 2")

        print("init done")

    def receive_new_frame(self, data_dict):
        if IN_DOMESTIC_SUITE:
            model_names = ["bot2", "bot3"]  # NOTE: Domestic Suite
        else:
            model_names = ["Go2Body", "bot2", "Arx5Gripper"]  # NOTE: Dog Room
            streaming_id_to_model_name = {
                1: "Go2Body",
                2: "Arx5Gripper",
                3: "bot2",
                4: "bot3",
            }
        rigid_body_list = data_dict["rigid_body_data"].rigid_body_list
        for i, rigid_body in enumerate(rigid_body_list):
            if rigid_body.tracking_valid:
                if IN_DOMESTIC_SUITE:
                    id_num = rigid_body.id_num
                    robot_pos = rigid_body.pos
                    robot_ori = rigid_body.rot
                    pose = np.array(
                        [
                            robot_pos[2],
                            robot_pos[0],
                            np.arctan2(robot_ori[1], robot_ori[3]) * 2,
                        ]
                    )
                    while pose[-1] < -np.pi:
                        pose[-1] += np.pi * 2
                    while pose[-1] > np.pi:
                        pose[-1] -= np.pi * 2
                    if "bot" in model_names[id_num - 1]:
                        self._rigid_body_data[model_names[id_num - 1]] = pose
                else:
                    rigid_body_name = streaming_id_to_model_name[rigid_body.id_num]
                    if "bot" not in rigid_body_name:
                        continue
                    robot_pos = rigid_body.pos
                    robot_ori = rigid_body.rot
                    pose = np.array(
                        [
                            robot_pos[2],
                            robot_pos[0],
                            np.arctan2(robot_ori[1], robot_ori[3]) * 2,
                        ]
                    )
                    pose[-1] += np.pi  # NOTE: JUST FOR Dog Room
                    while pose[-1] < -np.pi:
                        pose[-1] += np.pi * 2
                    while pose[-1] > np.pi:
                        pose[-1] -= np.pi * 2
                    self._rigid_body_data[rigid_body_name] = pose
        if self._save_to_shm:
            self._shared_memory_lock.acquire()
            self._shared_memory[:] = self._rigid_body_data[self._focus_bot_name]
            self._shared_memory_lock.release()

    @property
    def rigid_body_data(self):
        return self._rigid_body_data


if __name__ == "__main__":
    agent = MocapAgent()
    while True:
        print(agent._rigid_body_data, end="\r")
        time.sleep(1.0 / 120)
