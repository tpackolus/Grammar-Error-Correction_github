import os
import h5py

def check_hdf5_file(filepath):
    # Check if the file exists
    if not os.path.exists(filepath):
        return False, "File does not exist."
    
    # Check if the file size is greater than zero
    if os.path.getsize(filepath) == 0:
        return False, "File is empty."
    
    try:
        # Try to open the file using h5py
        with h5py.File(filepath, 'r') as f:
            # Print the contents of the file
            print("Contents of the HDF5 file:")
            for key in f.keys():
                print(f" - Key: {key}, Shape: {f[key].shape}, Type: {f[key].dtype}")
            
            # Check for the 'input' dataset in the file
            if 'input' not in f.keys():
                return False, "File does not contain the 'input' dataset."
            
            # Check if the 'input' dataset is not empty
            if f['input'].shape[0] == 0:
                return False, "'input' dataset is empty."
            
            # Optionally: Add more checks as needed
            
    except Exception as e:
        return False, f"Error while reading the file: {e}"
    
    return True, "File is valid."

# Check the aa.hdf5 file
is_valid, message = check_hdf5_file("/mnt/sharedhome/harahus/Grammar-Error-Correction_github/data/bb.hdf5")
print(message)
