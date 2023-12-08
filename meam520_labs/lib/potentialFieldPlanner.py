import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from time import perf_counter


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE
        '''
        d = 1.0
        if np.linalg.norm(current - target)**2 > d:
            att_f = -(current - target)/np.linalg.norm(current - target)
        else:
            att_f = -1*(current - target)

        att_f = -1 * (current - target)
        '''
        att_f = np.zeros((3, 1)) 
        #simple proportional controller using parabolic well approach
        k_att = 1.0
        att_f = k_att * (target - current)
        
        ## END STUDENT CODE
        return att_f.reshape(3,1)
    
    @staticmethod
    def get_unitvec_to_closest_point_on_box(current, obstacle):
        #Extract min and max points of the box
        box_min = obstacle[:3]
        box_max = obstacle[3:]

        #Find the closest point on the box to the current position
        closest_point = np.maximum(box_min, np.minimum(box_max, current))

        #Compute the vector from the current position to the closest point
        vector_to_closest_point = closest_point - current

        #normalize the vector to get the unit vector
        distance = np.linalg.norm(vector_to_closest_point)
        if distance == 0:
            return np.zeros_like(current)
        unitvec = vector_to_closest_point/distance

        return unitvec, distance

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        corresponding to element x_min y_min z_min x_max y_max z_max
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        #rep_f = np.zeros((3, 1)) 
        '''
        if unitvec is None:
            unitvec, distance = PotentialFieldPlanner.get_unitvec_to_closest_point_on_box(current, obstacle)
        else:
            distance = np.linalg.norm(unitvec)
        '''
        if np.all(unitvec == 0):
            # Convert current to nx3 array and obstacle to 1x6 array for the function call
            dist, unitvec = PotentialFieldPlanner.dist_point2box(np.array(current).reshape(-1,3), np.array(obstacle))
            
            # Extract the first (and only) elements as we have only one point and one box
            distance = dist[0]
            unitvec = unitvec[0]
        else:
            distance = np.linalg.norm(unitvec)

        k_rep = 0.1 # Repulsive force gain, adjust as needed
        safety_distance = 1.0 # Distance at which repulsive force starts

        rep_f = np.zeros((3, 1)) 
    
        if distance < safety_distance and distance != 0:
            # Calculate the repulsive force only if within the safety distance
            force_magnitude = k_rep * (1/distance - 1/safety_distance) * (1/distance**2)
            rep_f = force_magnitude * unitvec

        ## END STUDENT CODE
        #print("rep_f:", rep_f, " and shape: ", rep_f.shape)
        return rep_f.reshape(3,1)

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)
        
        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x7 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x7 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros((3, 7)) 

        for i in range(7): # Robot has 7 joints
            # Compute attractive force for the joint
            att_f = PotentialFieldPlanner.attractive_force(target[:, i], current[:,i])
            
            # Initialize repulsive force for the joint
            rep_f = np.zeros((3,1))

            # Compute repulsive force from each obstacle
            for obs in obstacle:
                rep_f += PotentialFieldPlanner.repulsive_force(obs, current[:,i])
            #print(rep_f)
            # Sum of attractive and repulsive forces
            joint_forces[:, i] = att_f.flatten() + rep_f.flatten()
        # print("joint_forces:", joint_forces)

        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on each joint 
        """
        '''
        ## STUDENT CODE STARTS HERE
        J = calcJacobian(q)
        #joint_torques = np.zeros((1, 7)) 
        joint_torques = np.zeros(7)
        
        for i in range(7):
            force_vector = joint_forces[:, i]
            joint_torques[i]  = np.dot(J[:3, i].T, force_vector)
        
        
        linear_forces = np.sum(joint_forces, axis=1).reshape(3,1)
        joint_torques = np.dot(J[0:3, :].T, linear_forces).flatten()
        ## END STUDENT CODE
        '''

        J = calcJacobian(q) 
        J_linear = J[:3,:] #3x7

        joint_torques = np.zeros((1,7))

        for i in range(7):
            Jv_i = np.zeros((3,7))
            Jv_i[:, :i+1] = J_linear[:, :i+1]
            torque_i = np.dot(Jv_i.T, joint_forces[:,i])
            joint_torques += torque_i
        '''
        for i in range(7):
            Jv_i = np.zeros((3,i+1))
            Jv_i[:, :i+1] = J_linear[:, :i+1]

            # torque_i =  np.dot(Jv_i.T, joint_forces[:, i])
            torque_i = Jv_i.T @ joint_forces[:, i] 
            #joint_torques += torque_i
            joint_torques[:,:i+1] += torque_i 
        ''' #This works!!!

        return joint_torques.flatten()

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        current_joint_positions, _ = PotentialFieldPlanner.fk.forward(q)
        current_joint_positions = current_joint_positions[:-1].T # Ignore end effector position
        # print("a",current_joint_positions)
        target_joint_positions, _ = PotentialFieldPlanner.fk.forward(target)
        target_joint_positions = target_joint_positions[:-1].T  
        # print("b",target_joint_positions)
        #Compute forces on each joint
        joint_forces = PotentialFieldPlanner.compute_forces(target_joint_positions, map_struct.obstacles, current_joint_positions)
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)

        #Compute gradient by normalizing joint_torques
        if np.linalg.norm(joint_torques) != 0:
            dq = joint_torques / np.linalg.norm(joint_torques)
        else:
            dq = np.zeros_like(joint_torques)
        ## END STUDENT CODE
        #dq = joint_torques
        dq = joint_torques 
        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """
        """
        q_path = np.array([]).reshape(0,7)

        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 

            # Termination Conditions
            if True: # TODO: check termination conditions
                break # exit the while loop if conditions are met!

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function 

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            
            ## END STUDENT CODE

        return q_path
        """

        q_path = [start]
        current_q = start
        step_size = 0.01 # Adjust step size as needed
        stuck_counter = 0
        
        
        for _ in range(self.max_steps):
        
            # Compute gradient
            dq = PotentialFieldPlanner.compute_gradient(current_q, goal, map_struct)
            
            if np.linalg.norm(dq) < self.min_step_size: # Check if gradient is too small
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            if stuck_counter > 10: # Threshold to determine if stuck at local minima
                # Generate random walk within joint limits

                dq = np.random.rand(7) - 0.5 # Random walk
                dq = dq / np.linalg.norm(dq) # Normalize
                new_q = current_q + step_size * dq

                new_q = np.maximum(new_q, self.lower) # Constrain new_q within joint_limits
                new_q = np.minimum(new_q, self.upper)

                dq = (new_q - current_q) / step_size # Update dq based on constrained new_q
                stuck_counter = 0

            else:
            
            # Update joint angles
                new_q = current_q + step_size * dq

            # Collision Check
            current_positions, _ = PotentialFieldPlanner.fk.forward(current_q)
            new_positions, _ = PotentialFieldPlanner.fk.forward(new_q)
            collision_detected = False
            
            for i in range(len(current_positions)): # check collision for all joint and end effector positions
                # Reshape the points to 2D arrays with shape (1,3)
                start_point = current_positions[i].reshape(1,3)
                end_point = new_positions[i].reshape(1,3)

                # Check for collision
                for obstacle in map_struct.obstacles:
                    if (True in detectCollision(start_point, end_point, obstacle)): # Collision detected

                        collision_detected = True
                        break # exit obstacle for loop if collision is detected

                if collision_detected:
                    break # Exit joint for loop if collision detected

            if not collision_detected: # No collision, update current configuration

                current_q = new_q
                q_path.append(current_q)
            
            #current_q = new_q
            #q_path.append(current_q)


            if PotentialFieldPlanner.q_distance(current_q, goal) < self.tol: # Check termination condition (close to goal)
                break
               
        return np.array(q_path)


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    '''
    map_struct = loadmap("../maps/emptyMap.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    #goal = np.array([0,1,0,-2,0,0,0])
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))
        print("Error is: ", error)

    # print("q path: ", q_path)
    '''
    starts = [np.array([0, -1, 0, -2, 0, 1.57, 0]),
          np.array([0, 0.4, 0, -2.5, 0, 2.7, 0.707]),
          np.array([0.5, 1, 0.5, -2, 0, 1.57, 0]),
          np.array([0, -0.2, 0, -2, 0, 1.57, 0]),
          np.array([0, -0.4, 0, -2, 0, 1.57, 0]),
          np.array([0, -0.4, 0, -2, 0, 1.57, 0]),
          np.array([0, 0.2, 0, -2, 0, 1.57, 0]),
          np.array([0, -1, 0.3, -1, 0, 0.57, 0]),
          np.array([0, 0.8, 0, -2, 0, 1.57, 0]),
          np.array([0, -1, 0, -2, 0, 1.57, 0])]
          
    
    goals = [np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7]),
         np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707]),
         np.array([1.5, 0.72, 1.5, -1.07, -0.57, 2.57, 1.7]),
         np.array([-1.2, 0.72, 0.6, -2.07, -0.57, 1.57, 0.7]),
         np.array([-1.4, 1.27, 1.27, -1.67, -1.57, 1.57, 1.7]),
         np.array([-1.6, 0.27, 1.27, -1.47, -0.57, 1.57, 0.75]),
         np.array([-1.1, -1.27, 1.27, -2.07, -0.57, 1.57, 0.7]),
         np.array([1.4, 1.27, 1.27, -2.07, -0.57, 1.57, 0.9]),
         np.array([1.2, -1.27, 1.27, -2.07, -0.57, 2.37, 0.7]),
         np.array([1.0, -1.47, 0.77, -1.37, -0.65, 1.37, 1.3])]
    
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])


    map_struct = loadmap("../maps/map1.txt")
    success = 0
    time_list = []
    for index in range(len(starts)):
        start = perf_counter()
        q_path = planner.plan(deepcopy(map_struct), deepcopy(starts[index]), deepcopy(goals[index]))
        for i in range(q_path.shape[0]):
            error = PotentialFieldPlanner.q_distance(q_path[i, :], goals[index])
            #print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))
            print("Error is: ", error)
            if error < 1:
                print("Success: path found ")
                success += 1
            
        stop = perf_counter()
        dt = stop - start
        time_list.append(dt)
         
        print("Potential Field Planner took {time:2.2f} sec. Path is.".format(time=dt))
        print(np.round(q_path,4))
    print("Success rate is ", success/len(starts))
    print("Average Time taken ", sum(time_list)/len(time_list))