import numpy as np
import pinocchio as pin

from scipy.spatial.transform import Rotation

"""
Wrapper for pinocchio, offers Forward & Inverse Kinematics & Dynamics for 7DOF Kinova gen3 robot
Feb 16, 2024

@sophielueth, @ziangcao
"""


def create_q_unlimited(q, joints_unlimited):
    """
    creates the requires joint configuration representation for pinocchio

    Args:
        q (np.ndarray of shape (7,)): joint configuration to change into pinocchio representation in rad
        joints_unlimited (list of bool, length 7): indicator whether joint corresponding to this position is unlimited

    Return:
        np.array of shape (7 + num_umlimited_joints, 1): correct joint configuration representation for pinocchio
    """
    q_pin = []
    for q_i, joint_unlimited in zip(q, joints_unlimited):
        if joint_unlimited:
            q_pin.append(np.cos(q_i))
            q_pin.append(np.sin(q_i))
        else:
            q_pin.append(q_i)

    return np.array(q_pin).reshape(-1, 1)


def inv_q_unlimited(q_pin, joints_unlimited):
    """
    inverses pinocchio's representation of unlimited joints (each cos, sin)

    Args:
        q_pin (np.ndarray of shape (7 + num_umlimited_joints, 1)): joint configuration representation for/from pinocchio
        joint_unlimited (list of bool, length 7): indicator whether joint corresponding to this position is unlimited

    Return:
        np.ndarray of shape (7,): joint configuration as in one rad value for each joint in (-pi; pi]
    """
    q = []

    i = 0
    for joint_unlimited in joints_unlimited:
        if joint_unlimited:
            q.append(np.arctan2(q_pin[i + 1], q_pin[i]))
            i += 2
        else:
            q.append(q_pin[i])
            i += 1

    return np.array(q)


def damped_pseudo_inverse(J, damp=1e-10):
    """
    Returns damped pseudo inverse of J, according to:

       min_{q_dot} (x_dot - J @ q_dot).T @ (x_dot - J @ q_dot) + damp * q_dot.T * q_dot
       q_dot = J.T @ (J @ J.T + damp*I).inv x_dot

    Is numerically more stable than the pseudo inverse in case of singularities.

    Args:
       J (np.ndarray of shape (6, q_dim)): robotic Jacobian matrix according to: x_dot = J @ q
       damp (float): damping coefficient
    Returns:
       np.ndarray of shape (q_dim, 6): damped_pseudo_inverse
    """
    return J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))


class Transform:
    def __init__(
        self, translation=np.zeros((3,)), rotation=Rotation.from_quat([1, 0, 0, 0])
    ):
        """
        Args:
            translation (np.ndarray of shape (3,)): Translation of Homogenous Transform
            rotation (Rotation): rotation of Homogenous Transform
        """
        self._translation = translation
        self._rotation = rotation

        self._matrix = np.zeros((4, 4))
        self._matrix[:3, :3] = rotation.as_matrix()
        self._matrix[:3, 3] = translation

    def to_matrix(self):
        return self._matrix

    def to_pos_quat(self):
        return self._translation, self._rotation.as_quat()


class KinovaPinModel:
    def __init__(self, urdf_path, tool_frame="tool_frame"):
        """
        Args:
            urdf_path: path to URDF file of Kinova Gen 3 7 DOF with unlimited joints 1, 3, 5, 7 (see self.joints_umlimited)
            tool_frame (str): the name of the frame used for default for FK & IK
        """

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()  # for algorithmic buffering
        self.ee_frame = tool_frame

        # # Set position limits for the robot
        # limited_indices = [False, False, True, False, False, True, False, False, True, False, False]
        # self.model.lowerPositionLimit[limited_indices] = np.array([-2.1, -2.36, -1.93])
        # self.model.upperPositionLimit[limited_indices] = np.array([2.1, 2.36, 1.93])

        # Set position limits for the robot
        limited_indices = [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
        ]
        self.model.lowerPositionLimit[limited_indices] = np.array([-2.1, -2.36, -1.93])
        self.model.upperPositionLimit[limited_indices] = np.array([2.1, 2.36, 1.93])

        self.joints_unlimited = [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
        ]  # whether rotational joint has position limits or not; NOTE: could be read out by model.joints[i].nq (nq=2 for unlimited joints); for some reason, only in the URDF file without the robotiq, joints 1, 3, 5, 7 are defined unlimited

    def forward_kinematics(self, q, frame=None):
        """
        Computes the homogenous transform at the specified joint for the given joint configuration.

        Args:
            q (np.ndarray of shape (7,)): joint configuration to compute forward kinematics for in rad
            frame (str): name of the frame to compute the transform for

        Return:
            Transform: homogenous transform at the end-effector
        """
        q_pin = self.to_q_pin(q)

        if frame is None:
            frame = self.ee_frame

        pin.framesForwardKinematics(self.model, self.data, q_pin)

        frame_id = self.model.getFrameId(frame)
        frame = self.data.oMf[frame_id]
        return Transform(
            translation=frame.translation, rotation=Rotation.from_matrix(frame.rotation)
        )

    def inverse_kinematics(
        self,
        des_trans,
        q=None,
        frame=None,
        pos_threshold=0.005,
        angle_threshold=5.0 * np.pi / 180,
        n_trials=7,
        dt=0.1,
    ):
        """
        Get IK joint configuration for desired pose of specified joint frame.

        Args:
            des_trans (Transform): desired frame transform for the frame specified via joint_ind
            q (np.ndarray of shape (7,)): joint start configuration, if applicable
            frame (str): name of the frame to compute the inverse kinematics for
            pos_threshold (float): in m
            angle_threshold (float): in rad
            n_trials (int):
            dt (float): in s, used as stepsize for gradient descent (Jacobian)

        Return:
            bool, : whether the inverse kinematics found a solution within the
                    thresholds
            np.ndarray of shape (7,) : best joint configuration found/the
                    first one to fulfill the requirement thresholds
        """
        damp = 1e-10
        success = False

        if frame is None:
            frame = self.ee_frame

        oMdes = pin.SE3(des_trans.to_matrix())
        frame_id = self.model.getFrameId(frame)

        if q is not None:
            q_pin = self.to_q_pin(q)
            # breakpoint()

        for n in range(n_trials):
            if q is None:
                q_pin = np.random.uniform(
                    self.model.lowerPositionLimit, self.model.upperPositionLimit
                )

            for i in range(800):
                # breakpoint()
                pin.framesForwardKinematics(self.model, self.data, q_pin)
                oMf = self.data.oMf[frame_id]
                dMf = oMdes.actInv(oMf)
                err = pin.log(dMf).vector

                if (np.linalg.norm(err[0:3]) < pos_threshold) and (
                    np.linalg.norm(err[3:6]) < angle_threshold
                ):
                    success = True
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q_pin, frame_id)
                v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q_pin = pin.integrate(self.model, q_pin, v * dt)
                q_pin = np.clip(
                    q_pin, self.model.lowerPositionLimit, self.model.upperPositionLimit
                )

            if success:
                best_q = np.array(q_pin)
                break
            else:
                # Save current solution
                best_q = np.array(q_pin)

        best_q = inv_q_unlimited(best_q, self.joints_unlimited)

        return success, best_q

    def forward_dynamics(self, q, q_dot, tau):
        """
        Computing joint accelerations for given joint configuration, joint velocity and joint torque.

        Args:
            q (np.ndarray of shape (7,)): joint configuration in rad
            q_dot (np.ndarray of shape (7,)): joint velocity in rad/s
            tau (np.ndarray of shape (7,)): joint torque in Nm

        Return:
            np.ndarray of shape (7,) : joint acceleration in rad/s^2
        """

        q_pin, q_dot_pin = self.to_q_pin(q, q_dot)

        q_dotdot = pin.aba(self.model, self.data, q_pin, q_dot_pin, tau)
        q_dotdot = q_dotdot[:7]

        return q_dotdot

    def inverse_dynamics(self, q, q_dot=None, q_dotdot=None):
        """
        Computing the necessary joint torques to achieve the given acceleration
        for joint configuration and velocity.

        Args:
            q (np.ndarray of shape (7,)): joint configuration in rad
            q_dot (np.ndarray of shape (7,)): joint velocity in rad/s
            q_dotdot (np.ndarray of shape (7,)): joint acceleration in rad/s^s

        Returns:
            np.ndarray of shape (7,): joint torque in Nm
        """
        if q_dot is None:
            q_dot = np.zeros((7,))
        if q_dotdot is None:
            q_dotdot = np.zeros((7,))

        q_pin, q_dot_pin, q_dotdot_pin = self.to_q_pin(q, q_dot, q_dotdot)

        tau = pin.rnea(self.model, self.data, q_pin, q_dot_pin, q_dotdot_pin)

        return tau

    def mass_matrix(self, q):
        """
        returns the 7x7 mass matrix for the given joint configuration

        Args:
            q (np.ndarray of shape (7,)): joint configuration in rad
        Returns:
            np.ndarray of shape (7,7): mass matrix for the gripperless robot
        """
        q_pin = self.to_q_pin(q)

        pin.crba(self.model, self.data, q_pin)
        return self.data.M[:7, :7]

    def coriolis_vector(self, q, q_dot):
        """
        returns coriolis vector for the given joint configuration & velocity

        Args:
            q (np.ndarray of shape (7,)): joint configuration in rad
            q_dot (np.ndarray of shape (7,)): joint velocity configuration in rad/s

        Returns:
            np.ndarray of shape (7,): coriolis vector for the gripperless robot
        """
        q_pin, q_dot_pin = self.to_q_pin(q, q_dot)

        pin.computeCoriolisMatrix(self.model, self.data, q_pin, q_dot_pin)
        return self.data.C[:7, :7] @ q_dot

    def gravity_vector(self, q):
        """
        computes torques needed to compensate gravity

        Args:
            q (np.ndarray of shape (7,)): joint configuration in rad

        Returns: np.ndarray of shape (7,): grav comp vector

        """
        q_pin = self.to_q_pin(q)
        pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        return self.data.g

    def jacobian(self, q, frame=None):
        """
        computes Jacobian for given joint configuration for a given Frame

        Args:
            q (np.ndarray of shape (7,)): joint configuration
            frame (str): The frame to compute the Jacobian for

        Returns:
            np.ndarray of shape (6,7)
        """
        q_pin = self.to_q_pin(q)

        if frame is None:
            frame = self.ee_frame

        frame_id = frame_id = self.model.getFrameId(frame)

        return pin.computeFrameJacobian(self.model, self.data, frame_id, pin.WORLD)[
            :6, :7
        ]

    def jacobian_dot(self, q, q_dot):
        """
        returns dJ/dt, with J being the Jacobian Matrix

        Args:
            q (np.ndarray of shape (7,)):
            q_dot (np.ndarray of shape (7,)):

        Returns:
            np.ndarray of shape (7,7): the time derivative of the Jacobian
        """
        q_pin, q_dot_pin = self.to_q_pin(q, q_dot)

        pin.computeJointJacobiansTimeVariation(self.model, self.data, q_pin, q_dot_pin)
        return pin.getFrameJacobianTimeVariation(self.model, self.data, 7, pin.WORLD)[
            :7, :7
        ]

    def to_q_pin(self, q, q_dot=None, q_dotdot=None):
        """
        transforms given (7,) shape np.ndarrays into suitable forms for pinocchio depending on whether the gripper is used
        """
        q_pin = create_q_unlimited(q, self.joints_unlimited)
        # breakpoint()
        if q_dot is not None:
            q_dot_pin = q_dot.reshape(-1, 1)
        if q_dotdot is not None:
            q_dotdot_pin = q_dotdot.reshape(-1, 1)

        if q_dot is None and q_dotdot is None:
            return q_pin
        elif q_dot is None:
            return q_pin, q_dotdot_pin
        elif q_dotdot is None:
            return q_pin, q_dot_pin
        else:
            return q_pin, q_dot_pin, q_dotdot_pin
