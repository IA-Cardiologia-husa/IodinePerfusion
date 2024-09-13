import pickle
import sys

def main():
	model_path = sys.argv[1]
	
	with open(model_path, 'rb') as f:
		model = pickle.load(f)
	if hasattr(model, 'best_estimator_'):
		model = model.best_estimator_

	linear_model = model[-1]
	scaler = model[-2]

	b = linear_model.intercept_

	coefs = []

	for c,s,m,name in zip(linear_model.coef_, scaler.scale_,scaler.mean_, scaler.get_feature_names_out()):
		coefs.append(c/s)
		print(name, c/s)
		b-=c*m/s
	print("Intercept", b)

if __name__ == '__main__':
	main()