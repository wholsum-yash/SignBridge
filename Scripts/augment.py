import random

import numpy as np
from numpy.random.bit_generator import SeedlessSeedSequence


# injecting noise
def add_spatial_noise(sequence, noise_range=(0.01, 0.03)):

    random_noise = np.random.uniform(*noise_range)  
    noise = np.random.normal(0, random_noise , sequence.shape)
    return sequence + noise


# temporal warp
def temporal_warp(sequence):
    seq_len = len(sequence)

    factor = np.random.uniform(0.9, 1.1)
    new_length = int(seq_len * factor)

    indices = np.linspace(0, seq_len -1, new_length).astype(int)
    warped = sequence[indices]

    # padding or trimming back to original length
    if(len(warped) < seq_len):
        pad = np.repeat(warped[-1][None, :], seq_len - len(warped), axis = 0)
        warped = np.vstack([warped, pad])

    else:
        warped = warped[:seq_len]

    return warped

# simulating random frame drop
def frame_drop(sequence, drop_range = (2,5)):
    sequence = sequence.copy()

    random_drop = np.random.randint(*drop_range)
    drop_indices = np.random.choice(len(sequence), random_drop, replace = False)

    for idx in drop_indices:
        sequence[idx] = 0 # simulating missing sequence

    return sequence


# simulating scale and shift
def scale_and_shift(sequence, scale_range = (0.85, 1.15), shift_range = (-0.1, 0.1)):
        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range, size = sequence.shape[1])

        return sequence * scale + shift

# random occulsion
def random_occulsion(sequence, prob=0.1):
    sequence = sequence.copy()

    mask = np.random.rand(*sequence.shape) < prob
    sequence[mask]  = 0

    return sequence
    
# simulating random rotation (-15 to +15 degrees)
def random_rotate(sequence, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    theta = np.radians(angle)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 2D rotation (x, y only, z stays the same)
    rotation_matrix = np.array([
                               [cos_theta, -sin_theta],
                               [sin_theta, cos_theta]
                           ])

    rotated = np.zeros_like(sequence)

    for t in range(sequence.shape[0]):
        frame = sequence[t].reshape(-1, 3)  # (42, 3)

        xy = frame[:, :2]  # take x, y
        z = frame[:, 2:]  # keep z

        xy_rotated = xy @ rotation_matrix.T

        frame_rotated = np.hstack([xy_rotated, z])

        rotated[t] = frame_rotated.flatten()

    return rotated

# Augment function (Main)
def augment_sequence(sequence):
    augmented = []

    # original
    augmented.append(sequence)

    # adding variations at random
    for aug in range(5):
        seq = sequence.copy()

        seq = add_spatial_noise(seq)
        seq = temporal_warp(seq)
        seq = frame_drop(seq)
        seq = scale_and_shift(seq)
        seq = random_occulsion(seq)
        
        if np.random.rand() < 0.3:
            seq = random_rotate(seq)

        augmented.append(seq)

    return augmented
