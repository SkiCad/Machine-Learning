import numpy as np

# Load data and labels into numpy arrays
data = ...  # shape = (number of samples, number of impedance-frequency pairs)
labels = ...  # shape = (number of samples,)

# Save numpy arrays as .npy files
np.save('skin_data.npy', data)
np.save('skin_labels.npy', labels)