import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
%matplotlib inline
import warnings 
warnings.filterwarnings('ignore')
defaultDataset = pd.read_csv ("c:/Users/eyiwu/Documents/ml-data.csv")
defaultDataset.head()
defaultDataset.Default_flag.value_counts()
sns.countplot(x='Default_flag',  data = defaultDataset, palette='hls')
plt.show()
defaultDataset.shape
x= defaultDataset[[ 'GDP', 'Rating_change', ]]
y= defaultDataset[['Default_flag']]
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.95, random_state=0)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix 
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test,y_pred))
y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score (y_test, y_pred_proba)
plt.plot(fpr,tpr,label='data 1,  auc=' + str(auc))
plt.legend(loc=4)
plt.show()