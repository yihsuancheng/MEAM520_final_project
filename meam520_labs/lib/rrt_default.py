import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK

fk = FK()

def steer(from_node, to_node, step_size=0.1):
    """
    Steers from from_node towards to_node with a given step_size.

    :param from_node: The node in the tree from which to steer.
    :param to_node: The target node to steer towards.
    :param step_size: The maximum distance to move towards the to_node.
    :return: A new node moved from from_node towards to_node by step_size.
    """
    vector = to_node - from_node
    distance = np.linalg.norm(vector)
    if distance < step_size:
        return to_node
    else:
        return from_node + (vector / distance) * step_size

def isCollisionFree(fk, nearestNode, newPoint, obstacles):
    joint_positions, _ = fk.forward(nearestNode)
    new_joint_positions, _ = fk.forward(newPoint)
    
    for obs in obstacles:
        if (True in detectCollision(joint_positions, new_joint_positions, obs)):
            return False
    return True

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    
    # initialize path
    
    #path = [start]
    tree = {tuple(start): None}
    foundPath = False
    iteration = 0
    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    max_iter = 1000
    goal_sampling_rate = 0.2
    

    while not foundPath and iteration < max_iter:

        if np.random.rand() < goal_sampling_rate:
            randPoint = goal
        else:
            randPoint = lowerLim + np.random.random(size=lowerLim.shape) * (upperLim - lowerLim)

        nearest_node = min(tree, key=lambda node: np.linalg.norm(np.array(node) - randPoint))
        newPoint = steer(nearest_node, randPoint, step_size=0.5)

        if isCollisionFree(fk, nearest_node, newPoint, map.obstacles):
            tree[tuple(newPoint)] = tuple(nearest_node)
            # Check if newPoint can directly connect to goal
            if isCollisionFree(fk, newPoint, goal, map.obstacles):
                tree[tuple(goal)] = tuple(newPoint)
                foundPath = True
                break

            '''
            if np.linalg.norm(newPoint - goal) < 0.1:
                foundPath = True
                break
            '''
        iteration += 1
    
    if foundPath:
        # Backtrack to find the path
        path = []
        current_node = tuple(goal)
        while current_node is not None:
            path.append(current_node)
            current_node = tree[current_node]
        return np.array(path[::-1]) #reverse the path
    
    else:
        return np.array([])  # Return empty array if no path is found
    
if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    print("Path: ", path)
