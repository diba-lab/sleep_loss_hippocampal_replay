# Analysis code readme
The `sleep_deprivation` folder contains all the Jupyter notebooks used to generate the figures for this paper. The `DataPaths` folder includes all the workhouse, custom-written Python code referenced in the notebooks.

# Steps to get up and running
### 1) Clone this repository and Neuropy
- Neuropy: https://github.com/diba-lab/NeuroPy
- You will need to create a conda/mamba environment per the `environment.yml` file in the NeuroPy repository.

### 2) Download Processed Data
- This is housed in `ProcessedData` folder hosted in the main repository.

### 3) Adjust search paths
- the `DataPaths/subjects.py` file contains several references to data file locations that need to be adjusted to run the code.
- Change the `GroupData.path` property to the location where your processed data downloaded in step 2 resides.
- This should allow you to generate almost all the figure panels used in this paper, with the exception of Figure 1B, C, E, F, and a few others.
- These panels require raw electrophysiological binary files which are extremely large and can be made available upon request. Contact Kamran Diba (kdiba [at] umich [dot] edu).

### 4) Run a notebook
- To start, open `sleep_deprivation\figures\sd_figure1_bs.ipynb` (bs stands for bootstrapping, since hierarchical bootstrapping was used for the majority of statistical analyses).
- The top cell will contain the lines `sys.path.extend(['/home/nkinsky/Documents/GitHub/NeuroPy'])` and a similar line for the `DataPaths` folder.  Change these to match the location of your local NeuroPy repository and `DataPaths` folder.
- Run the code!

# Notes on file naming and figure panel referencing
The code used to generate figures is located in the `sd_figurex_bs.ipynb` files. Extended Data figure panels related to each figure are in the the `sd_figurx_supp.ipynb` files.  Note that, in some cases, extended data figure panels were produced in a main figure notebook. All figure panels are noted in Jupyter cells within each notebook
