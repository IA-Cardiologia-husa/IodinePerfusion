# Iodine Perfusion model training

This code provides the training and evaluation of the models for the research paper *SPECTRAL CT REFERENCE VALUES FOR MYOCARDIAL STATIC RESTING PERFUSION. EXPLORING SEX DIFFERENCES THROUGH MACHINE LEARNING*. To execute the code one needs to create a Python environment with the following libraries installed:
- numpy
- pandas
- openpyxl
- xlsxwriter
- luigi
- scikit-learn
- xgboost

The code can be run with the following command:

	python -m luigi --module KoopaML AllTasks --local-scheduler
	
This will create several folders. ML algorithms will be placed as .pickle files in *models* folder. Validation results can be seen in the *report* folder. The original data to train the models is not provided, but a *fakedata.xlsx* file is used instead to simulate the training. The actual models described in the research article, and the evaluation report, can be found in the folders *final models* and *final report* respectively.

Data ingestion can be modified in the file *user_data_utils.py*, ML algorithms can be specified in the file *user_MLmodels_info.py*, and the workflow of training/validation can be specified in *user_Workflow_info.py*.

To explore the coefficients of one of the trained models, execute the following script:

	python coefs.py model.pickle

To obtain a summary of statistics of the segmentary differences in iodine density, execute the following script:

	python iodine_distribution.py data.xlsx

The files *final_coefs.txt* and *final_distribution.txt* contain the output of these scripts with the original data.
