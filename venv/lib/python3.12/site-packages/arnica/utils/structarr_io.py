import numpy as np




def da2sa(data_dict: dict)-> np.array:
    """copy a dictionary of numpy arrays to a Structured array"""
# Example dictionary
# data_dict = {
#     'a': np.array([1, 2, 3]),
#     'b': np.array([4.5, 5.5, 6.5]),
#     'c': np.array(['x', 'y', 'z'])
# }

    # Determine structured array dtype
    dtype = [(key, arr.dtype) for key, arr in data_dict.items()]

    # Convert dictionary to structured array
    structured_array = np.zeros(len(next(iter(data_dict.values()))), dtype=dtype)

    # Fill structured array
    for key in data_dict:
        structured_array[key] = data_dict[key]
    return structured_array

def sa2da(structured_array:np.array)-> dict:
    """Convert a structured array to a dict of arrays"""
    
    return {name: structured_array[name] for name in structured_array.dtype.names}

data_dict = {
    'a': np.array([1, 2, 3]),
    'b': np.array([4.5, 5.5, 6.5]),
    'c': np.array(['x', 'y', 'z'])
}
print(data_dict)
print(da2sa(data_dict))
print(sa2da(da2sa(data_dict)))