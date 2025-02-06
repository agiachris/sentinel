import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np


def retime(waypoints, interval=0.001, max_vel=150, max_accel=200, max_decel=150):
    n, d = waypoints.shape
    path = ta.SplineInterpolator(np.linspace(0, 1, n), waypoints)
    vlim = np.array([[-max_vel, max_vel]] * d)
    alim = np.array([[-max_decel, max_accel]] * d)
    interp = constraint.DiscretizationType.Interpolation
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim, discretization_scheme=interp)
    instance = algo.TOPPRA([pc_vel, pc_acc], path)
    joint_traj = instance.compute_trajectory(0, 0)
    duration = joint_traj.get_duration()
    ts = np.linspace(0, duration, int(duration / interval))
    return joint_traj.eval(ts), joint_traj.evald(ts), joint_traj.evaldd(ts)
