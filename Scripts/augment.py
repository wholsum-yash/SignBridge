import random

import numpy as np


# injecting noise
def add_noise(sequence, noise_level=0.005):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


# adding frame skipping
def temporal_drop(sequence, drop_ratio=0.05):
    seq_len = sequence.shape[0]
    keep = int(seq_len * (1 - drop_ratio))

    indices = sorted(random.sample(range(seq_len), keep))
    new_seq = sequence[indices]

    # padding back to original length
    while len(new_seq) < seq_len:
        new_seq = np.vstack([new_seq, new_seq[-1]])

    return new_seq


# simulating distance
def scale(sequence, scale_range=(0.9, 1.1)):
    factor = random.uniform(*scale_range)

    return sequence * factor


# Simulating camera movement
def shift(sequence, shift_range=0.005):
    shift_val = random.uniform(-shift_range, shift_range)

    return sequence + shift_val


# simulating random rotation (-15 to +15 degrees)
def random_rotate(sequence, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    theta = np.radians(angle)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 2D rotation (x, y only, z stays the same)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated = np.zeros_like(sequence)

    for t in range(sequence.shape[0]):
        frame = sequence[t].reshape(-1, 3)  # (42, 3)

        xy = frame[:, :2]  # take x, y
        z = frame[:, 2:]  # keep z

        xy_rotated = xy @ rotation_matrix.T

        frame_rotated = np.hstack([xy_rotated, z])

        rotated[t] = frame_rotated.flatten()

    return rotated


# Mixup augmentation
def mixup(sequence, other_sequence, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    return lam * sequence + (1 - lam) * other_sequence


# Augment function (Main)
def augment_sequence(sequence):
    augmented = []

    # original
    augmented.append(sequence)

    # adding variations
    augmented.append(add_noise(sequence))
    augmented.append(temporal_drop(sequence))
    augmented.append(scale(sequence))
    augmented.append(shift(sequence))
    augmented.append(random_rotate(sequence))

    return augmented
