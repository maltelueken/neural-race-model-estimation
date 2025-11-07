import h5py


def save_hdf5(filename, data_dict):
    with h5py.File(filename, 'w') as file:
        for key, val in data_dict.items():
            if val is not None:
                _ = file.create_dataset(key, data=val)
            else:
                _ = file.create_dataset(key, data=h5py.Empty("f"))


def load_hdf5(filename):
    data_dict = {}

    with h5py.File(filename, 'r') as file:
        for key, val in file.items():
            if val.shape is not None:
                data_dict[key] = val[()]
            else:
                data_dict[key] = None

    return data_dict
