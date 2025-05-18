import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


'''
HDF5 is a popular file format designed to store large amounts of numerical data in a way that is both hierarchical and efficient. You can think an HDF5 file like a file system with in a file. There are "groups" and "dataset", this structure allows you to
    - organize data in a logical hierarchy
    - read or write only the required data subset, instead of loading everything into the memory
    - easily handle large, multidimensional arrays that may not fit into RAM
    - attach meta data to each dataset for self-describing files
'''    


# Create a directory to store output images, if you want to save image files
os.makedirs('saved_images', exist_ok=True)

with h5py.File('mnist.hdf5', 'r') as f:
    # Convert the keys to a list
    hdf5_keys = list(f.keys())
    print("Datasets in the file:", hdf5_keys)
    
    # Get references to the dataset objects
    images = f[hdf5_keys[0]]
    labels = f[hdf5_keys[1]]
    
    # Read *all* images into memory as a NumPy array
    all_images = images[:]       # or images[()] is also fine
    all_labels = labels[:]
    
print("all_images type:", type(all_images))
print("all_images shape:", all_images.shape)
print("all_labels shape:", all_labels.shape)

# For example, now you can iterate through all_images in memory:
for i in range(len(all_images)):
    img = all_images[i]
    label = all_labels[i]
    
    # OPTIONAL: save each image as a PNG for later viewing
    plt.imsave(f"saved_images/image_{i}_label_{label}.png", img, cmap='gray')



