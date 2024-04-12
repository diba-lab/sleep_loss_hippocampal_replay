import numpy as np
import h5py
from pathlib import Path
import pandas as pd

source_dir = Path("/home/bapung/Documents")
file = source_dir / "Test2.mat"
# one way
# arrays = {}
# f = h5py.File(file, "r")
# for k, v in f.items():
#     arrays[k] = np.array(v)
# df = pd.read_hdf(file)
data = h5py.File(file)



# for var in data:
#     name = var[0]
#     data = var[1]
#     print "Name ", name  # Name
#     if type(data) is h5py.Dataset:
#         # If DataSet pull the associated Data
#         # If not a dataset, you may need to access the element sub-items
#         value = data.value
#         print "Value", value  # NumPy Array / Value

# another way
# data_within = data["results0"]["trialinfo"]
