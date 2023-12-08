import numpy as np
from lib.calculateFK import FK
from scipy.optimize import minimize
#from core.interfaces import ArmController
#import rospy

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    fk = FK()
    joints, T0e = fk.forward(q_in)
    T_array = fk.compute_T_array(q_in)
    R_matrix = [T[0:3,0:3] for T in T_array]
    R01, R02, R03, R04, R05, R06, R0e = R_matrix
    
    Pe = T_array[-1][0:3,3]
    z_axis = np.array([0,0,1])

    Jv = []
    for i, R in enumerate(R_matrix):
        if i == 0:
            Jv.append(np.cross(z_axis.T, (Pe - joints[i])))
        elif i > 0 and i < 7:
            Jv.append(np.cross(R_matrix[i-1][0:3,2], (Pe - joints[i])))
            
    Jw = [np.array([0,0,1])]
    Jw.extend([np.matmul(R, np.array([0,0,1])) for R in R_matrix[0:6]])
    J = np.vstack([np.array(Jv).T, np.array(Jw).T])


    #J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE

    return J

def cost_function(q_in):
    J = calcJacobian(q_in)
    w = np.linalg.det(J @ J.T)
    return w  # Minimize -w to find configurations where w is close to 0



if __name__ == '__main__':
    '''q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
    print(q.shape)'''
        # Initial guess
    '''rospy.init_node('jacobian')
    arm = ArmController()
    joint_limits = arm.get_joint_limits()
    print("Joint limits: ", arm.joint_limits())
    print("Get joint limits: ", arm.get_joint_limits())
    position_lower = joint_limits.position_lower
    position_upper = joint_limits.position_upper
    bounds = list(zip(position_lower, position_upper))
    print("joint limit bounds: ", bounds)
    result = minimize(cost_function, np.random.rand(7), bounds=bounds)

     Singular configuration
    q_in_singular = result.x
    print("singular configurations: ", q_in_singular)
    w = cost_function(q_in_singular)
    print(w)'''
    
    '''q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    w = cost_function(q)
    print("manipulability: ", w)'''