import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK
from time import perf_counter
'''
Code for RRT-Connect
'''

fk = FK()

def check_entire_path_collision_free(fk, path, obstacles):
    collision_indices = []
    for i in range(len(path) - 1):
        if not isCollisionFree(fk, path[i], path[i+1], obstacles):
            collision_indices.append(i)
    return True if len(collision_indices) == 0 else False

def isjointCollisionFree(fk, newPoint, obstacles):
    joint_positions, _ = fk.forward(newPoint)
    for obs in obstacles:
        for i in range(7):
            # check whether joint i with collide with obstacles
            if (True in detectCollision(joint_positions[i].reshape(1,3), joint_positions[i+1].reshape(1,3), obs)):
                return True
    return False

def newPointCollisionFree(fk, nearestNode, newPoint, obstacles):
    interval = np.array(newPoint) - np.array(nearestNode)
    interval_range = 10
    flag = []
    dq = interval/interval_range
    for i in range(interval_range):
        q = nearestNode + i * dq
        if isjointCollisionFree(fk, q, obstacles):
            flag.append(q)
    return True if len(flag) != 0 else False # if len of list is not 0 that means there is collision


def isCollisionFree(fk, nearestNode, newPoint, obstacles):
    joint_positions, _ = fk.forward(nearestNode)
    new_joint_positions, _ = fk.forward(newPoint)
    
    for obs in obstacles:
        if (True in detectCollision(joint_positions, joint_positions, obs)):
            return False
        
        elif (True in detectCollision(new_joint_positions, new_joint_positions, obs)):
            return False
    
    for obs in obstacles:
        if (True in detectCollision(joint_positions, new_joint_positions, obs)):
            return False
        
    return True

def newConfig(tree, point, step_size):
    nearestNode = min(tree, key=lambda node: np.linalg.norm(np.array(node) - np.array(point)))
    direction = np.array(point) - np.array(nearestNode)
    length = np.linalg.norm(direction)
    norm_vec = direction / length
    distance = min(step_size, length)

    newPoint = np.array(nearestNode) + distance * norm_vec
    return newPoint

def extend2(tree, nearestNode, newPoint):
    tree[tuple(newPoint)] = tuple(nearestNode)
    return tuple(newPoint)


def extend(tree, point, fk, map, step_size):
    nearestNode = min(tree, key=lambda node: np.linalg.norm(np.array(node) - np.array(point)))
    direction = np.array(point) - np.array(nearestNode)
    length = np.linalg.norm(direction)
    norm_vec = direction / length
    distance = min(step_size, length)

    newPoint = np.array(nearestNode) + distance * norm_vec
    
    if isCollisionFree(fk, nearestNode, newPoint, map.obstacles):
        tree[tuple(newPoint)] = tuple(nearestNode)
        return tuple(newPoint)
    
    return None


def connect(tree, point, fk, map, step_size):
    extended = True
    while extended:
        newPoint = extend(tree, point, fk, map, step_size)
        if newPoint is None:
            return False, None
        
        # Check if the new point is close enough to the goal or can connect directly
        
        if np.linalg.norm(np.array(newPoint) - np.array(point)) <= step_size:
            if isCollisionFree(fk, point, newPoint, map.obstacles):
                #tree[tuple(goal)] = tuple(newPoint)
                    return True, tuple(newPoint)
        
        extended = np.linalg.norm(np.array(newPoint) - np.array(point)) > step_size
    # quits the loop when the newPoint and point is close
    return True, tuple(newPoint)

def build_path(tree_a, tree_b, connect_point_a, connect_point_b, is_tree_a_start):
    path = []

    # Backtrack from connect_point_a to start
    while connect_point_a is not None:
        path.append(connect_point_a)
        connect_point_a = tree_a[connect_point_a]
    path = path[::-1] # Reverse the path

    # Backtrack from connect_point_b to goal
    connect_point_b = tree_b[connect_point_b] # Skip the connecting point as it's already included
    while connect_point_b is not None:
        path.append(connect_point_b)
        connect_point_b = tree_b[connect_point_b]
    
    if not is_tree_a_start:
        return np.array(path[::-1])
    
    return np.array(path)

def rrt(map, start, goal):
    
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    tree_a = {tuple(start): None} #Tree from start
    tree_b = {tuple(goal): None}  #Tree from goal

    max_iter = 10000
    step_size = 0.1
    is_tree_a_start = True # Flag to identify which tree is the start tree
    
    for i in range(max_iter):
        # Sample random point
        randPoint = np.array([random.uniform(lowerLim[0], upperLim[0]), 
                           random.uniform(lowerLim[1], upperLim[1]), 
                           random.uniform(lowerLim[2], upperLim[2]), 
                           random.uniform(lowerLim[3], upperLim[3]), 
                           random.uniform(lowerLim[4], upperLim[4]), 
                           random.uniform(lowerLim[5], upperLim[5]), 
                           random.uniform(lowerLim[6], upperLim[6])])
        
        #randPoint = np.random.uniform(low=lowerLim, high=upperLim)
        #randPoint = lowerLim + np.random.random(size=lowerLim.shape) * (upperLim - lowerLim)

        # Extend tree_a towards random point
        nearestNode = min(tree_a, key=lambda node: np.linalg.norm(np.array(node) - np.array(randPoint)))
        #newPoint_a = extend(tree_a, randPoint, fk, map, step_size)
        newPoint_a = newConfig(tree_a, randPoint, step_size)
        '''
        if newPoint_a is None:
            continue
        '''
        if any(newPoint_a < lowerLim) or any(newPoint_a > upperLim):
            continue

        elif isjointCollisionFree(fk, newPoint_a, map.obstacles):
            continue
        
        elif newPointCollisionFree(fk, nearestNode, newPoint_a, map.obstacles):
            continue
        
        newPoint_a = extend2(tree_a, nearestNode, newPoint_a)

        if newPoint_a is not None:
            # Try to connect tree_b to newPoint_a
            success, final_point = connect(tree_b, newPoint_a, fk, map, step_size)
            if success:
                path =  build_path(tree_a, tree_b, newPoint_a, final_point, is_tree_a_start)
                
                collision_free = check_entire_path_collision_free(fk, path, map.obstacles)

                print(collision_free)
            
                if collision_free:
                    return path
                
        # Swap trees
        tree_a, tree_b = tree_b, tree_a
        is_tree_a_start = not is_tree_a_start 

    return np.array([]) # Return empty path if no connection made
    
if __name__ == '__main__':
    map_struct = loadmap("../maps/map2.txt")
    
    '''
    start = np.array([0,-1,0,-2,0,1.57,0])
    #start = np.array([0, 0.4, 0, -2.5, 0, 2.7, 0.707])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    #goal = np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])
    #goal =  np.array([0,-1,0,-2,0,1.57,0])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    print("Path: ", path)
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

    success = 0
    time_list = []
    for index in range(len(starts)):
        start = perf_counter()
        path = rrt(deepcopy(map_struct), deepcopy(starts[index]), deepcopy(goals[index]))
        if len(path) != 0:
            success += 1
            print("Success: Path Found")
            
        stop = perf_counter()
        dt = stop - start
        time_list.append(dt)
        print("RRT took {time:2.2f} sec. Path is.".format(time=dt))
        print(np.round(path,4))
    print("Success rate is ", success/len(starts))
    print("Average Time taken ", sum(time_list)/len(time_list))
    