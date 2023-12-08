from lib.calculateFK import FK
from core.interfaces import ArmController

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import Delaunay

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
 ]

def sample_joint_angles(limits, num_samples=10000):
    sampled_joint_angles = []
    for i in limits:
        sampled_joint_angles.append(np.random.uniform(i['lower'], i['upper'], num_samples))
    return np.array(sampled_joint_angles)

joint_samples = sample_joint_angles(limits, 3000)

joint_positions = []
end_effector_positions = []
for combination in np.nditer(joint_samples, flags = ["external_loop"], order = "F"):
    joint_pos, T0e = fk.forward(combination)
    joint_positions.extend(joint_pos)
    end_effector_positions.append(T0e[:3,3])
joint_positions = np.array(joint_positions)
end_effector_positions = np.array(end_effector_positions)

"""
# Define bounding box for end effector positions
x_min, y_min, z_min = end_effector_positions.min(axis=0)
x_max, y_max, z_max = end_effector_positions.max(axis=0)

# Generate candidate points inside bounding box
num_points = 5000  # Number of candidate points
random_points = np.vstack((np.random.uniform(x_min, x_max, num_points),
                           np.random.uniform(y_min, y_max, num_points),
                           np.random.uniform(z_min, z_max, num_points))).T

# Use Delaunay triangulation to find points inside the boundary
tri = Delaunay(end_effector_positions)
mask = tri.find_simplex(random_points) >= 0
inside_points = random_points[mask]"""


# TODO: create plot(s) which visualize the reachable workspace of the Panda arm,
# accounting for the joint limits.
#
# We've included some very basic plotting commands below, but you can find
# more functionality at https://matplotlib.org/stable/index.html

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# TODO: update this with real results
#ax.scatter(1,1,1) # plot the point (1,1,1)

#ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:,2], c = "b", marker="o", label = "Joint Positions")

#inside_points = np.array(inside_points)
#ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], c='g', marker='.', label='Inside Workspace', alpha=0.5)

ax.scatter(end_effector_positions[:, 0],end_effector_positions[:, 1], end_effector_positions[:, 2], c = "g", marker= ".", label = "End Effector Pose" )
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
#ax.set_title("Reachable Workspace of the Panda Robot Arm")
ax.legend()
plt.show()
