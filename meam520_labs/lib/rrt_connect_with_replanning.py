import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK
'''
Code for RRT-Connect
'''

fk = FK()

'''
def sampleCollisionPoints(fk, nearestNode, newPoint, interpolate):
    sampled_positions = []
    increment = (newPoint - nearestNode)/(1 / interpolate)
    interpolate_iter = 0
    while (interpolate_iter != 1):
        nearestNode = nearestNode + increment
        sampled_positions.append(nearestNode)
        interpolate_iter += interpolate
    sampled_positions.append(newPoint)
    num_samples = len(sampled_positions)
    return sampled_positions, num_samples

def isCollisionFree(fk, nearestNode, newPoint, obstacles):
    sampled_positions, num_samples = sampleCollisionPoints(fk, nearestNode, newPoint, interpolate=0.2)

    sampled_joint_positions = []

    for config in sampled_positions:
        current_joint_positions, _ = fk.forward(config)
        sampled_joint_positions.append(current_joint_positions)

    for i in range(num_samples - 1):
        for obs in obstacles:
            if (True in detectCollision(sampled_joint_positions[i], sampled_joint_positions[i+1], obs)):
                #print("Robot collides with obstacles")
                return False
    return True
        
'''

def check_entire_path_collision_free(fk, path, obstacles):
    collision_indices = []
    for i in range(len(path) - 1):
        if not isCollisionFree(fk, path[i], path[i+1], obstacles):
            collision_indices.append(i)
    if len(collision_indices) == 0:
        return True
    return False
'''
def replan_segment(fk, tree_a, tree_b, start, end, map, step_size, max_attempt=100):
    # Initialize new trees for the replanning segment
    local_tree_a = {tuple(start): None}
    local_tree_b = {tuple(end): None}

    for i in range(max_attempt):
        randPoint = start + np.random.random(size=start.shape) * (end - start)

        # Extend a tree towards the random point
        newPoint_a = extend(local_tree_a, randPoint, fk, map, step_size)
        if newPoint_a is not None:
            success, final_point = connect(local_tree_b, newPoint_a, fk, map, step_size)
            if success:
                return build_path(local_tree_a, local_tree_b, newPoint_a, final_point, True)
'''
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
        if np.linalg.norm(np.array(newPoint) - np.array(point)) <= 0.001:
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

    max_iter = 5000
    step_size = 0.1
    max_total_replans = 50
    total_replans = 0
    is_tree_a_start = True # Flag to identify which tree is the start tree
    
    for i in range(max_iter):
        # Sample random point
        randPoint = np.random.uniform(low=lowerLim, high=upperLim)
        #randPoint = lowerLim + np.random.random(size=lowerLim.shape) * (upperLim - lowerLim)

        # Extend tree_a towards random point
        newPoint_a = extend(tree_a, randPoint, fk, map, step_size)
        if newPoint_a is not None:
            # Try to connect tree_b to newPoint_a
            success, final_point = connect(tree_b, newPoint_a, fk, map, step_size)
            if success:
                path =  build_path(tree_a, tree_b, newPoint_a, final_point, is_tree_a_start)
                #return path
                collision_free = check_entire_path_collision_free(fk, path, map.obstacles)

                print(collision_free)
                '''
                while not collision_free and total_replans < max_total_replans:
                    for index in collision_indices:
                        start_segment = path[index]
                        end_segment = path[index + 1] if index + 1 < len(path) else goal

                        alternative_segment = replan_segment(fk, tree_a, tree_b, start_segment, end_segment, map, step_size)
                        if alternative_segment is not None:
                            path = np.concatenate((path[:index], alternative_segment, path[index+2:]))
                            total_replans += 1
                            break # Break the loop and recheck the path

                    collision_free, collision_indices = check_entire_path_collision_free(fk, path, map.obstacles)
                '''
                if collision_free:
                    return path

        # Swap trees
        tree_a, tree_b = tree_b, tree_a
        is_tree_a_start = not is_tree_a_start 

    return np.array([]) # Return empty path if no connection made
    
if __name__ == '__main__':
    map_struct = loadmap("../maps/map3.txt")
    
    #start = np.array([0,-1,0,-2,0,1.57,0])
    start = np.array([0, 0.4, 0, -2.5, 0, 2.7, 0.707])
    #goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    goal = np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])
    #goal =  np.array([0,-1,0,-2,0,1.57,0])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    print("Path: ", path)
