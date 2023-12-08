import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    #dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    #v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    #omega_in = omega_in.reshape((3,1))

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    J = calcJacobian(q_in)
    desired_velocity = np.vstack((v_in, omega_in))

    valid_indices = ~np.isnan(desired_velocity)
    desired_velocity = desired_velocity[valid_indices].reshape(-1,1)
    J = J[valid_indices[:,0], :]
    J_pinv = np.linalg.pinv(J)
    #psuedo_inverse = np.dot(J.T, np.linal   g.inv(np.dot(J,J.T)))
    #dq = np.dot(np.linalg.pinv(J), desired_velocity)
    #dq, _, _, _ = np.linalg.lstsq(J, desired_velocity, rcond=None)
    #dq = np.dot(J_pinv, desired_velocity)

    dq = np.dot(J_pinv, desired_velocity)
    N = np.eye(J.shape[1]) - np.dot(J_pinv, J)
    null = np.dot(N, b)

    return dq + null

