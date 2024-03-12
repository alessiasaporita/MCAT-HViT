import pickle
import h5py

"""
This function serializes the object save_object and writes the resulting byte stream to the file-like object file writer
"""
def save_pkl(filename, save_object): 
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

"""
This function reads a byte stream from the file-like loader file and deserializes it into a Python object
"""
def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


"""
saving data to an HDF5 file 

output_path: The path to the HDF5 file where the data will be saved.
asset_dict: A dictionary where keys are dataset names and values are NumPy arrays containing the data to be saved.
attr_dict: An optional dictionary containing attributes to be added to the datasets.
mode: The mode in which the HDF5 file is opened ('a' for append, 'w' for write, etc.).
"""
def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path