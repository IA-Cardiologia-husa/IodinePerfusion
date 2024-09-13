import pandas as pd
import numpy as np
import sys

def main():
	data_path = sys.argv[1]
	
	df = pd.read_excel(data_path)
	
	print(f"Mean Iodine Density for all patients:", df[f'Mean Iodine Density'].mean())
	print(f"Standard Deviation:", df[f'Mean Iodine Density'].std())
	print()
	
	for segment in range(1,18,1):
		diff = df[f'Segment {segment} Iodine Density'] - df['Mean Iodine Density']
		print(f"Segment {segment:02}: {diff.mean(): .3g}, stdev: {diff.std(): .3g}")

if __name__ == '__main__':
	main()