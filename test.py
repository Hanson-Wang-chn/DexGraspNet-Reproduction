import numpy as np

# ----------Calculation methods from the paper----------
# Rotation matrix
def rotation_matrix(axis, theta):
    """
    Generate a rotation matrix around a specified axis
    axis: Rotation axis, 3D vector
    theta: Rotation angle in radians
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
                     [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]])

# Skew-symmetric matrix
def skew_symmetric_matrix(x):
    return np.array([[    0, -x[2],  x[1]],
                     [ x[2],     0, -x[0]],
                     [-x[1],  x[0],     0]])

# Compute E_fc
def compute_E_fc(contact_points):
    """
    contact_points: List of contact points (3D vectors)
    """
    n = len(contact_points)
    I = np.identity(3)
    G_list = []
    for x_i in contact_points:
        G_i = np.vstack((I, skew_symmetric_matrix(x_i)))
        G_list.append(G_i)
    G = np.hstack(G_list)
    E_fc = np.linalg.norm(G, ord=2)
    return E_fc

# Compute E_dis
def compute_E_dis(contact_points, object_surface):
    """
    contact_points: List of contact points
    object_surface: Object surface point cloud
    """
    E_dis = 0
    for x_i in contact_points:
        distances = np.linalg.norm(object_surface - x_i, axis=1) # Minimum distance
        min_distance = np.min(distances)
        E_dis += min_distance
    return E_dis

# Compute E_pen
def compute_E_pen(hand_surface_points, object_surface):
    """
    hand_surface_points: Hand surface point cloud
    object_surface: Object surface point cloud
    """
    E_pen = 0
    for v in hand_surface_points:
        # Determine if hand points are inside the object
        distances = np.linalg.norm(object_surface - v, axis=1)
        min_distance = np.min(distances)
        if min_distance < penetration_threshold: # Simplified to distance below threshold
            E_pen += (penetration_threshold - min_distance)
    return E_pen

# Compute E_s_pen
def compute_E_s_pen(hand_surface_points):
    """
    hand_surface_points: Hand surface point cloud
    """
    E_s_pen = 0
    for i, p in enumerate(hand_surface_points):
        for j, q in enumerate(hand_surface_points):
            if i != j:
                distance = np.linalg.norm(p - q)
                if distance < self_collision_threshold:
                    E_s_pen += (self_collision_threshold - distance)
    return E_s_pen

# Compute E_joints
def compute_E_joints(theta, theta_min, theta_max):
    """
    theta: Current joint angles
    theta_min: Minimum joint angles
    theta_max: Maximum joint angles
    """
    E_joints = 0
    for i in range(len(theta)):
        if theta[i] < theta_min[i]:
            E_joints += (theta_min[i] - theta[i])
        elif theta[i] > theta_max[i]:
            E_joints += (theta[i] - theta_max[i])
    return E_joints


# ----------Set example parameters----------
penetration_threshold = 0.01
self_collision_threshold = 0.01

# Consider random points on a unit sphere
object_surface = np.random.rand(100, 3)

# Pose initialization
T = np.array([0.0, 0.0, 0.0]) # Translation vector
R = np.identity(3) # Rotation matrix
theta = np.zeros(22) # ShadowHand has 22 joint angles

# Joint angle limits
theta_min = -np.pi * np.ones(22)
theta_max = np.pi * np.ones(22)

# Contact points initialization
contact_points = np.array([
    [0.05, 0.0, 0.0],
    [0.0, 0.05, 0.0],
    [-0.05, 0.0, 0.0],
    [0.0, -0.05, 0.0]
])

# Fixed hand surface point cloud, also using random points on a unit sphere
hand_surface_points = np.random.rand(100, 3)

# Weights given in the paper
w_dis = 100
w_pen = 100
w_s_pen = 10
w_joints = 1


# ----------Optimization process----------
learning_rate = 0.01
num_iterations = 1000

for iteration in range(num_iterations):
    E_fc = compute_E_fc(contact_points)
    E_dis = compute_E_dis(contact_points, object_surface)
    E_pen = compute_E_pen(hand_surface_points, object_surface)
    E_s_pen = compute_E_s_pen(hand_surface_points)
    E_joints = compute_E_joints(theta, theta_min, theta_max)
    
    E_total = E_fc + w_dis * E_dis + w_pen * E_pen + w_s_pen * E_s_pen + w_joints * E_joints
    
    # Demonstrate with random gradients
    grad_T = np.random.randn(3) * 0.01
    grad_R = np.random.randn(3, 3) * 0.01
    grad_theta = np.random.randn(22) * 0.01
    
    # Update parameters
    T -= learning_rate * grad_T
    R -= learning_rate * grad_R
    theta -= learning_rate * grad_theta
    
    # Ensure the rotation matrix is orthogonal
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

print("In the tuple g=(T, R, theta):")
print("T=", T)
print("R=", R)
print("theta=", theta)
