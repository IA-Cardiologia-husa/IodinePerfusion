import numpy as np
import pandas as pd
import sklearn.metrics as sk_m
import matplotlib.pyplot as plt


class aucroc():
	def __init__(self):
		self.name = 'aucroc'
		self.optimization_sign = 1
		self.variance = None


	def __call__(self, df, df_train=None):
		m = (df['True Label']==0).sum()
		n = (df['True Label']==1).sum()
		auc = sk_m.roc_auc_score(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),df.loc[df['True Label'].notnull(), 'Prediction'])
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		self.variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		self.std_error = np.sqrt(self.variance)
		return auc

	def plot(self, df, results, ax, score_name):
		roc_divisions = 1001
		fpr_va = np.linspace(0,1,roc_divisions)
		tpr_va = np.zeros(roc_divisions)
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		for rep in df['Repetition'].unique():
			for fold in df['Fold'].unique():
				true_label = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'True Label'].astype(bool).values
				pred_prob = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'Prediction'].values
				fpr, tpr, thresholds = sk_m.roc_curve(true_label,pred_prob)
				tpr_va += np.interp(fpr_va, fpr, tpr)
		tpr_va = tpr_va / (len(df['Repetition'].unique())*len(df['Fold'].unique()))


		plt.plot(fpr_va, tpr_va, lw=2, alpha=1, color=self.cmap(self.color_index) , label = f'{score_name}: AUC ={results["avg_aucroc"]:1.2f} ({results["aucroc_95ci_low"]:1.2f}-{results["aucroc_95ci_high"]:1.2f})' )
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])

		ax.set_xlabel('1-specificity', fontsize = 15)
		ax.set_ylabel('sensitivity', fontsize = 15)

		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		fig.savefig(name)
		return


class aucpr():
	def __init__(self):
		self.name = 'aucpr'
		self.optimization_sign = 1
		self.variance = None

	def __call__(self, df, df_train=None):
		m = (df['True Label']==0).sum()
		n = (df['True Label']==1).sum()
		auc = sk_m.average_precision_score(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),df.loc[df['True Label'].notnull(), 'Prediction'])
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		self.variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		self.std_error = np.sqrt(self.variance)
		return auc

	def plot(self, df, results, ax, score_name):
		pr_divisions = 1001
		prec_va = np.linspace(0,1,pr_divisions)
		recall_va = np.zeros(pr_divisions)
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		for rep in df['Repetition'].unique():
			for fold in df['Fold'].unique():
				true_label = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'True Label'].astype(bool).values
				pred_prob = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'Prediction'].values
				prec, recall, thresholds = sk_m.precision_recall_curve(true_label,pred_prob)

				recall_va += np.interp(prec_va, prec, recall)
		recall_va = recall_va / (len(df['Repetition'].unique())*len(df['Fold'].unique()))

		plt.plot(recall_va, prec_va, lw=2, alpha=1, color=self.cmap(self.color_index) , label = f'{score_name}: AUC ={results["avg_aucpr"]:1.2f} ({results["aucpr_95ci_low"]:1.2f}-{results["aucpr_95ci_high"]:1.2f})' )
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])

		ax.set_xlabel('1-specificity', fontsize = 15)
		ax.set_ylabel('sensitivity', fontsize = 15)

		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="upper right", fontsize = 10)
		fig.savefig(name)
		return

class rmse():
	def __init__(self):
		self.name = 'rmse'
		self.optimization_sign = -1
		self.variance = None
		self.max_x = None
		self.min_x = None
		self.max_y = None
		self.min_y = None
		self.index = 0

	def __call__(self, df, df_train=None):
		diff = df.loc[df['True Label'].notnull(), 'True Label'] - df.loc[df['True Label'].notnull(), 'Prediction']
		rmse = np.sqrt((diff**2).mean())

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return rmse

	def plot(self, df, results, ax, score_name):
		nx = self.index % self.sidex
		ny = self.index // self.sidex
		ax[ny, nx].scatter(df.loc[df['True Label'].notnull(), 'True Label'], df.loc[df['True Label'].notnull(), 'Prediction'],
							color=self.cmap(self.index%10), label = f'{score_name}: RMSE ={results["avg_rmse"]:1.2f} ({results["rmse_95ci_low"]:1.2f}-{results["rmse_95ci_high"]:1.2f})')
		ax[ny, nx].plot([0, df.loc[df['True Label'].notnull(), 'True Label'].max()], [0, df.loc[df['True Label'].notnull(), 'True Label'].max()], c='black', linestyle=':')
		ax[ny, nx].legend(loc="lower right", fontsize = 10)
		self.index+=1

	def figure(self, n, cmap = "tab10"):
		self.sidex = np.ceil(np.sqrt(n)).astype(int)
		self.sidey = np.ceil(n/self.sidex).astype(int)
		fig, ax = plt.subplots(self.sidey, self.sidex, figsize=(5*self.sidex,5*self.sidey), squeeze = False)

		for nx in range(self.sidex):
			for ny in range(self.sidey):
				ax[ny, nx].set_xlabel('True Value', fontsize = 15)
				ax[ny, nx].set_ylabel('Predicted Value', fontsize = 15)

		self.index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		fig.savefig(name)
		return


class r2():
	def __init__(self):
		self.name = 'mse_r2'
		self.optimization_sign = -1
		self.variance = None

	def __call__(self, df, df_train=None):
		diff = df.loc[df['True Label'].notnull(), 'True Label'] - df.loc[df['True Label'].notnull(), 'Prediction']

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return (diff**2).mean()

	def plot(self, df, results, ax, score_name):
		var_tl = df['True Label'].var()
		
		avg_r2 = 1- results["avg_mse_r2"]/var_tl
		r2_95ci_low = 1- results["mse_r2_95ci_low"]/var_tl
		r2_95ci_high = 1- results["mse_r2_95ci_high"]/var_tl

		ax.barh(f'{score_name}', avg_r2, color=self.cmap(self.color_index),
			   label = f'{score_name}: R2 ={avg_r2:.3g} ({r2_95ci_low:.3g}-{r2_95ci_high:.3g})')
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		ax.set_xlim([-0.05, 1.05])

		fig.savefig(name)
		return



metrics_list = {'aucroc': aucroc,
				'aucpr':aucpr,
				'rmse':rmse,
				'r2':r2}
