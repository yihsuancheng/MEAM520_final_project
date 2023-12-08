#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple


def loadmap(filename):
    """
    :param filename: string with the location of the map file
    :return: map struct with boundary and obstacles element
                map.obstacles [Nx6] array of the obstacle boundaries
                map.boundary [6] array of the map boundary
    """
    obstacles = []
    with open(filename, 'r') as reader:
        # Read file line by line
        line = reader.readline()
        while line != '':  # The EOF char is an empty string
            line = reader.readline()
            # Check to see if first character is b for boundary or block
            if len(line) > 0 and line[0] == 'b':
                words = line.split()
                # Check if block our boundary
                if words[0] == "block":
                    # Append to obstacles array or set to obstacles array
                    if len(obstacles) == 0:
                        obstacles = np.array([[float(words[i]) for i in range(1, len(words))]])
                    else:
                        obstacles = np.append(obstacles, np.array([[float(words[i]) for i in range(1, len(words))]]), axis=0)
    # Returns the map in a struct
    MyStruct = namedtuple("map", "obstacles")
    return MyStruct(obstacles = obstacles)

if __name__ == "__main__":
    
    mystruct = loadmap("../maps/map1.txt")

    #isCollided = np.ones(1)
    #print(isCollided)
    
    print(mystruct.obstacles)
    for i in mystruct.obstacles:
        print(np.array(i))
    print(mystruct.obstacles[0])
    ##for i in mystruct:
    #    print(i)
    
    '''
    #print(np.minimum([3,3,3], [5,4,0]))
    #print(np.array([[0, 2, 2], [2, 2, 2], [4, 0, 2]]))
    boxMin = np.array([1, 1, 1])
    boxMax = np.array([3, 3, 3])
    points = np.array([[0, 2, 2], [2, 2, 2], [4, 0, 2]])

    dx = np.amax(np.vstack([boxMin[0] - points[:, 0], points[:, 0] - boxMax[0], np.zeros(points.shape[0])]).T, 1)
    dy = np.amax(np.vstack([boxMin[1] - points[:, 1], points[:, 1] - boxMax[1], np.zeros(points.shape[0])]).T, 1)
    dz = np.amax(np.vstack([boxMin[2] - points[:, 2], points[:, 2] - boxMax[2], np.zeros(points.shape[0])]).T, 1)

    print(np.vstack([boxMin[0] - points[:, 0], points[:, 0] - boxMax[0], np.zeros(points.shape[0])]).T)
    print("dx:", dx)
    print("dy:", dy)
    print("dz:", dz)
    distances = np.vstack([dx, dy, dz]).T
    dist = np.linalg.norm(distances, axis=1)
    print(distances)
    print(dist)
    print(dist[0])
    # Figure out the signs
    boxCenter = boxMin*0.5 + boxMax*0.5
    
    signs = np.sign(boxCenter-points)

    # Calculate unit vector and replace with
    unit = distances / dist[:, np.newaxis] * signs
    print(unit)
    print(unit[0])
    
    joint_forces = np.array([
    [1, 2, 3, 4, 5, 6, 7],  # x-components of forces for each joint
    [8, 9, 10, 11, 12, 13, 14],  # y-components
    [15, 16, 17, 18, 19, 20, 21]  # z-components
    ])

    # Summing over axis=1
    sum_forces = np.sum(joint_forces, axis=1)

    print("Summed Forces:", sum_forces)
    '''

