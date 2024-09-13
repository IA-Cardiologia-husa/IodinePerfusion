# Dictionary of variables and formal names for the descriptive excel tables
# If left empty, all the variables will be included, using the column name of the dataframe
# If not empty, only the variables in the dictionary will be used for the descriptive tables,
# they will have the same order as the dictionary, and will use the value provided in the 
# dictionary as the name
#
# Example:
#
# dict_var =	{'Name' : 'Name'
#				 'G': 'Gender'
#				 'H': 'Height'
#				 'W': 'Weight',
#				 'BMI': 'Body Mass Index'}

dict_var = {}

for v in [	'Age',
			'AF History',
			'Sex (0: Woman, 1: Man)',
			'BMI',
			'Hypertension',
			'Diabetes',
			'Dyslipidaemia ',
			'Obesity',
			'Contrast Dose (mL)',
			'Blood Pool Iodine Density',
			'Myocardium Volume (mL)',
			'Smoking History',
			'Active Smoking',
			'Acquisition Type (0: Step-and-shoot, 1: Helical)',
			'Phase (0: Diastole, 1:Systole)',
			'Radiation Exposure, mA·s',
			'Radiation Exposure Time, ms',
			'X-Ray Current, mA',
			'Acquisition Duration',
			'CTDIvol',
			'Delay between tracker and acquisition']:
	dict_var[v] = v