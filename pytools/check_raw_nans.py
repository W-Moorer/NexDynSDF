
import struct
import numpy as np
import sys

def check_file(filepath):
    print(f"Checking {filepath}...")
    with open(filepath, 'rb') as f:
        # Read header
        grid_res = struct.unpack('i', f.read(4))[0]
        bbox_min = struct.unpack('fff', f.read(12))
        bbox_max = struct.unpack('fff', f.read(12))
        
        print(f"Resolution: {grid_res}")
        
        # Read data
        num_voxels = grid_res * grid_res * grid_res
        data_bytes = f.read(num_voxels * 4)
        grid_data = np.frombuffer(data_bytes, dtype=np.float32)
        
        # Check for NaNs and Infs
        nans = np.isnan(grid_data)
        infs = np.isinf(grid_data)
        
        num_nans = np.sum(nans)
        num_infs = np.sum(infs)
        
        print(f"Total voxels: {num_voxels}")
        print(f"NaNs: {num_nans} ({num_nans/num_voxels*100:.4f}%)")
        print(f"Infs: {num_infs} ({num_infs/num_voxels*100:.4f}%)")
        
        if num_nans > 0:
            print("First 10 NaNs indices:", np.where(nans)[0][:10])
            
        if num_infs > 0:
            print("First 10 Infs indices:", np.where(infs)[0][:10])
            
        # Range
        valid_data = grid_data[~(nans | infs)]
        if len(valid_data) > 0:
            print(f"Valid Data Range: [{np.min(valid_data)}, {np.max(valid_data)}]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_raw_nans.py <filepath>")
        sys.exit(1)
    check_file(sys.argv[1])
