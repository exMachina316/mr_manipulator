import pickle
import numpy as np
import random

# Load the dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

augmented_data = []
augmented_labels = []

def add_noise(sample, noise_level=0.02):
    """Add small Gaussian noise to landmarks."""
    noise = np.random.normal(0, noise_level, sample.shape)
    return sample + noise

def mirror_landmarks(sample):
    """Mirror the landmarks horizontally (flipping along x-axis)."""
    mirrored = sample.copy()
    for i in range(0, len(mirrored), 2):
        mirrored[i] = 1 - mirrored[i]  # Flip x-coordinate (assuming normalized [0,1] range)
    return mirrored

def jitter_landmarks(sample, jitter_factor=0.03):
    """Apply slight random shifts to landmarks."""
    jitter = np.random.uniform(-jitter_factor, jitter_factor, sample.shape)
    return np.clip(sample + jitter, 0, 1)  # Ensure values stay within [0,1] range

# Apply augmentations
for sample, label in zip(data, labels):
    augmented_data.append(sample)
    augmented_labels.append(label)

    # Apply transformations
    augmented_data.append(add_noise(sample))
    augmented_labels.append(label)

    augmented_data.append(mirror_landmarks(sample))
    augmented_labels.append(label)

    augmented_data.append(jitter_landmarks(sample))
    augmented_labels.append(label)

# Shuffle augmented dataset
combined = list(zip(augmented_data, augmented_labels))
random.shuffle(combined)
augmented_data, augmented_labels = zip(*combined)

# Save the new dataset
with open('augmented_data.pickle', 'wb') as f:
    pickle.dump({'data': augmented_data, 'labels': augmented_labels}, f)

print(f"Original dataset size: {len(data)}, Augmented dataset size: {len(augmented_data)}")
print("Augmented dataset saved as 'augmented_data.pickle'")
