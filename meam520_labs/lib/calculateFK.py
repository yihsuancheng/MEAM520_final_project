import numpy as np
from math import pi, cos, sin
import math

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        '''self.z1, self.z2, self.z3 = 0.141, 0.192, 0.195
        self.z4, self.z5, self.z6, self.z7 = 0.121, 0.0825, -0.051, -0.159
        self.x1, self.x2, self.x3, self.x4 = 0.0825, 0.125, 0.259, 0.088
        self.y1 = 0.015'''

    def get_transformation_matrix(self, angle, d, a, alpha):
        return np.array([
            [cos(angle), -sin(angle)*cos(alpha), sin(angle)*sin(alpha), a*cos(angle)],
            [sin(angle), cos(angle)*cos(alpha), -cos(angle)*sin(alpha), a*sin(angle)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
     #   A_W_0 = self.get_transformation_matrix(0, self.z1, 0, 0)
        A_0_1 = self.get_transformation_matrix(q[0], 0.192+0.142, 0, -pi/2)
        A_1_2 = self.get_transformation_matrix(q[1], 0, 0, pi/2)
        A_2_3 = self.get_transformation_matrix(q[2], 0.195+0.121, 0.0825, pi/2)
        A_3_4 = self.get_transformation_matrix(q[3] + pi, 0, 0.0825, pi/2)
        A_4_5 = self.get_transformation_matrix(q[4], 0.125+0.259, 0, -pi/2)  
        A_5_6 = self.get_transformation_matrix(q[5] - pi, 0, 0.088, pi/2)
        A_6_7 = self.get_transformation_matrix(q[6]-pi/4, 0.051+0.159, 0, 0)
        
        T0e =  np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(A_0_1, A_1_2),A_2_3), A_3_4), 
                      A_4_5), A_5_6), A_6_7)


        #jointPositions = np.zeros((8,3))
        #T0e = np.identity(4)'''

        T_1 = A_0_1
        T_2 = np.dot(A_0_1,A_1_2)
        T_3 = np.dot(T_2, A_2_3)
        T_4 = np.dot(T_3, A_3_4)
        T_5 = np.dot(T_4, A_4_5)
        T_6 = np.dot(T_5, A_5_6)
        T_7 = np.dot(T_6, A_6_7)

        T_array = [T_1, T_2, T_3, T_4, T_5, T_6, T_7]

        P_1 = np.dot(T_1, np.array([0,0.192,0,1]))
        P_2 = np.dot(T_2, np.array([0,0,0,1]))
        P_3 = np.dot(T_3, np.array([-0.0825, -0.121, 0, 1]))
        P_4 = np.dot(T_4, np.array([-0.0825, 0, 0, 1]))
        P_5 = np.dot(T_5, np.array([0, 0.259, 0, 1]))
        P_6 = np.dot(T_6, np.array([-0.088, -0.015, 0, 1]))
        P_7 = np.dot(T_7, np.array([0,0,-0.159,1]))
        P_8 = np.dot(T0e, np.array([0,0,0,1]))

        jointPositions = np.array([
            
            [P_1[0], P_1[1], P_1[2]],
            [P_2[0], P_2[1], P_2[2]],
            [P_3[0], P_3[1], P_3[2]],
            [P_4[0], P_4[1], P_4[2]],
            [P_5[0], P_5[1], P_5[2]],
            [P_6[0], P_6[1], P_6[2]],
            [P_7[0], P_7[1], P_7[2]],
            [P_8[0], P_8[1], P_8[2]]
            
        ])


        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_T_array(self, q):
        A_0_1 = self.get_transformation_matrix(q[0], 0.192+0.142, 0, -pi/2)
        A_1_2 = self.get_transformation_matrix(q[1], 0, 0, pi/2)
        A_2_3 = self.get_transformation_matrix(q[2], 0.195+0.121, 0.0825, pi/2)
        A_3_4 = self.get_transformation_matrix(q[3] + pi, 0, 0.0825, pi/2)
        A_4_5 = self.get_transformation_matrix(q[4], 0.125+0.259, 0, -pi/2)  
        A_5_6 = self.get_transformation_matrix(q[5] - pi, 0, 0.088, pi/2)
        A_6_7 = self.get_transformation_matrix(q[6]-pi/4, 0.051+0.159, 0, 0)
        
        T0e =  np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(A_0_1, A_1_2),A_2_3), A_3_4), 
                      A_4_5), A_5_6), A_6_7)


        #jointPositions = np.zeros((8,3))
        #T0e = np.identity(4)'''

        T_1 = A_0_1
        T_2 = np.dot(A_0_1,A_1_2)
        T_3 = np.dot(T_2, A_2_3)
        T_4 = np.dot(T_3, A_3_4)
        T_5 = np.dot(T_4, A_4_5)
        T_6 = np.dot(T_5, A_5_6)
        T_7 = np.dot(T_6, A_6_7)

        T_array = [T_1, T_2, T_3, T_4, T_5, T_6, T_7]
        return T_array

    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
    print(joint_positions.shape, T0e.shape)
    print(joint_positions[0].shape)
    print(joint_positions[0])
   
