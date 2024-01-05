import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import load_breast_cancer

from xgboost import XGBRegressor

from utils import get_data


X_train, X_test, y_train, y_test = get_data()

model = XGBRegressor()
model.fit(X_train, y_train)

y_binary = np.where(y_test > np.mean(y_test), 1, 0)
y_pred_proba = model.predict(X_test)

fpr, tpr, _ = roc_curve(y_binary, y_pred_proba)
auc = roc_auc_score(y_binary, y_pred_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: XGB')
plt.legend(loc='lower right')
plt.savefig('xgb.png')

print(f'Model AUC: {auc:.2f}')
