import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define frequency range and number of frequency values
freq_min = 1
freq_max = 100
num_freqs = 50

# Generate random impedance values for each frequency
freqs = np.linspace(freq_min, freq_max, num_freqs)
impedances = np.random.uniform(low=10, high=200, size=(num_freqs,))

# Define mean and standard deviation for normal and abnormal skin
normal_mean = impedances.mean()
normal_std = impedances.std()
abnormal_mean = impedances.mean() + 30
abnormal_std = impedances.std() * 1.5

# Generate synthetic data and labels
num_samples = 1000
num_abnormal = int(num_samples * 0.3)
num_normal = num_samples - num_abnormal

normal_data = np.random.normal(loc=normal_mean, scale=normal_std, size=(num_normal, num_freqs))
normal_labels = np.zeros((num_normal,))

abnormal_data = np.random.normal(loc=abnormal_mean, scale=abnormal_std, size=(num_abnormal, num_freqs))
abnormal_labels = np.ones((num_abnormal,))

data = np.vstack((normal_data, abnormal_data))
labels = np.concatenate((normal_labels, abnormal_labels))

# Shuffle data and labels
perm = np.random.permutation(num_samples)
data = data[perm]
labels = labels[perm]

# Save data and labels as .npy files
np.save('skin_data.npy', data)
np.save('skin_labels.npy', labels)
