import numpy as np


def flip_position(position):
    flipped_pos = np.concatenate((
        np.array([position[0]]),  # x
        np.array([-1 * position[1]]),  # y
        np.array([position[2]]),  # z

        np.array([position[3]]),  # w
        np.array([-1 * position[4]]),  # x
        np.array([position[5]]),  # y
        np.array([-1 * position[6]]),  # z

        # flip chest
        np.array([-1 * position[7]]),
        np.array([position[8]]),
        np.array([-1 * position[9]]),

        # flip neck
        np.array([-1 * position[10]]),
        np.array([position[11]]),
        np.array([-1 * position[12]]),

        # flip shoulders and elbow
        np.array([-1 * position[17]]),
        np.array([1 * position[18]]),
        np.array([-1 * position[19]]),
        np.array([position[20]]),  # elbow

        np.array([-1 * position[13]]),
        np.array([1 * position[14]]),
        np.array([-1 * position[15]]),
        np.array([position[16]]),  # elbow

        # flip hip knee and ankle
        # position[27:35],
        np.array([-1 * position[28]]),  # hip
        np.array([1 * position[29]]),  # hip
        np.array([-1 * position[30]]),  # hip
        np.array([position[31]]),  # knee
        np.array([-1 * position[32]]),  # ankle
        np.array([1 * position[33]]),  # ankle
        np.array([-1 * position[34]]),  # ankle

        # position[20:27],
        np.array([-1 * position[21]]),  # hip
        np.array([1 * position[22]]),  # hip
        np.array([-1 * position[23]]),  # hip
        np.array([position[24]]),  # knee
        np.array([-1 * position[25]]),  # ankle
        np.array([1 * position[26]]),  # ankle
        np.array([-1 * position[27]]),  # ankle
    ))

    return flipped_pos


def flip_velocity(velocity):
    flipped_velocity = np.concatenate((
        # xyz
        np.array([velocity[0]]),
        np.array([-1 * velocity[1]]),
        np.array([velocity[2]]),

        # rot x y z
        np.array([-1 * velocity[3]]),
        np.array([velocity[4]]),
        np.array([-1 * velocity[5]]),

        # flip chest
        np.array([-1 * velocity[6]]),
        np.array([velocity[7]]),
        np.array([-1 * velocity[8]]),

        # flip neck
        np.array([-1 * velocity[9]]),
        np.array([velocity[10]]),
        np.array([-1 * velocity[11]]),

        # flip shoulders and elbow
        np.array([-1 * velocity[16]]),
        np.array([1 * velocity[17]]),
        np.array([-1 * velocity[18]]),
        np.array([velocity[19]]),  # elbow

        np.array([-1 * velocity[12]]),
        np.array([1 * velocity[13]]),
        np.array([-1 * velocity[14]]),
        np.array([velocity[15]]),  # elbow

        # flip hip knee and ankle
        np.array([-1 * velocity[27]]),  # hip
        np.array([1 * velocity[28]]),  # hip
        np.array([-1 * velocity[29]]),  # hip
        np.array([velocity[30]]),  # knee
        np.array([-1 * velocity[31]]),  # ankle
        np.array([1 * velocity[32]]),  # ankle
        np.array([-1 * velocity[33]]),  # ankle

        np.array([-1 * velocity[20]]),  # hip
        np.array([1 * velocity[21]]),  # hip
        np.array([-1 * velocity[22]]),  # hip
        np.array([velocity[23]]),  # knee
        np.array([-1 * velocity[24]]),  # ankle
        np.array([1 * velocity[25]]),  # ankle
        np.array([-1 * velocity[26]]),  # ankle
    ))
    return flipped_velocity

def flip_action(action):
    joint_action = action[[0, 1, 2, 3, 4, 5,
                           10, 11, 12, 13,
                           6, 7, 8, 9,
                           21, 22, 23, 24, 25, 26, 27,
                           14, 15, 16, 17, 18, 19, 20]]

    flip = np.array([-1, 1, -1,  # chest
                     -1, 1, -1,  # neck
                     -1, 1, -1, 1,  # shoulder and elbow
                     -1, 1, -1, 1,  # shoulder and elbow
                     -1, 1, -1, 1, -1, 1, -1,  # hip knee ankle
                     -1, 1, -1, 1, -1, 1, -1, ])  # hip knee ankle

    joint_action = joint_action * flip
    return joint_action
