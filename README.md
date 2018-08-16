# Baseline
Run the baseline.py for baseline model.

# Overlapping
Please run the overlapping.py to run the overlapping model. Change the file path of overlap300/500/900 to change the overlapping rate.
# pca
Please run the pca.py to run the pca model.
# downsample
Please run the downsample.py to run the downsample model. The downsample should be run after overlapping() function. which is already in the code. (Dont need to edit the code)
# stateful
Please run the stateful.py to run the stateful model. The stateful model is based on the downsampled data.

# Dataprocessing
This folder includes the raw file collected from Smartisan and OnePlus. Run nolapping.ipynb to create orginal training data. Run overlapping.ipynb to generate overlapping training data. In the overlapping.ipynb, you could define the overlapping rate.

# Data visulisation tool
Run testtrain.py to visualise the sensor measurement from the raw log file to check if there is missing value in the raw file. 
