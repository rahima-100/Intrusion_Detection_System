# Intrusion Detection System

### Project Overview
This system is an intrusion detection system by using machine learning algorithms. It is comparing 5 classification models and assessing the performance of the models. Then it utilizes the outperforming model for the prediction of the network (either normal or anomaly).

### Tools
- Excel
- Python - Notebook Jupyter

### Installing and calling the necessary libraries
```python
!pip install mlxtend
!pip install -U scikit-learn
!pip install -U scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
```

### Data uploading

Uploading the Train and the Test dataset from the computer to the notejupyter environment

```python
Trained_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Trained_data.csv')
Tested_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Tested_data.csv')
```

```python
Trained_Data
```
![Trained_Data](https://github.com/user-attachments/assets/f30783a3-d9b7-4dea-bf73-9bafb6f4a3b5)


```python
Tested_Data
```
![Test_data](https://github.com/user-attachments/assets/75be11bd-8979-4985-9af0-ef8d263e9910)


### Data Preprocessing
```python
Results = set(Trained_Data['class'].values)
print(Results,end=" ")
```
![anomaly](https://github.com/user-attachments/assets/349a43d4-374c-4c9c-adf7-c3c299dd19b9)

#### Creation of attack_state column
```python
Trained_attack = Trained_Data['class'].map(lambda a: 0 if a == 'normal' else 1)
Tested_attack = Tested_Data['class'].map(lambda a: 0 if a == 'normal' else 1)

Trained_Data['attack_state'] = Trained_attack
Tested_Data['attack_state'] = Tested_attack
```
![attack_state](https://github.com/user-attachments/assets/b8d4a397-7e72-4a56-9ca7-ec98bc360665)

### Box Plotting for the Trained and Test Data
Trained Boxplot for checking outliers
```python
Trained_Data.plot(kind='box', subplots=True, layout=(8, 5), figsize=(20, 40))
plt.show()
```
![Trained_Boxplot](https://github.com/user-attachments/assets/5f59aaef-1ac4-42ea-9a0e-025e51de970e)

Tested Boxplot for checking outlier
```python
Tested_Data.plot(kind='box', subplots=True, layout=(8, 5), figsize=(20, 40))
plt.show()
```
![Test_Boxplot](https://github.com/user-attachments/assets/06179e7c-53eb-47bf-82cb-1bc7fabad9f0)

### Data Encoding
```python
## Data Encoding for Trained_data
Trained_Data = pd.get_dummies(Trained_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

## Data Encoding for Tested_data
Tested_Data = pd.get_dummies(Tested_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

LE = LabelEncoder()
attack_LE= LabelEncoder()
Trained_Data['class'] = attack_LE.fit_transform(Trained_Data["class"])
Tested_Data['class'] = attack_LE.fit_transform(Tested_Data["class"])
```

### Data Splitting
```python
### Data Splitting
X_train = Trained_Data.drop('class', axis = 1)
X_train = Trained_Data.drop('attack_state', axis = 1)

X_test = Tested_Data.drop('class', axis = 1)
X_test = Tested_Data.drop('attack_state', axis = 1)


Y_train = Trained_Data['attack_state']
Y_test = Tested_Data['attack_state']
```


### Data Scaling
```python
X_train_train,X_test_train ,Y_train_train,Y_test_train = train_test_split(X_train, Y_train, test_size= 0.25 , random_state=42)
X_train_test,X_test_test,Y_train_test,Y_test_test = train_test_split(X_test, Y_test, test_size= 0.25 , random_state=42)

### Data Scaling
Ro_scaler = RobustScaler()
X_train_train = Ro_scaler.fit_transform(X_train_train) 
X_test_train= Ro_scaler.transform(X_test_train)
X_train_test = Ro_scaler.fit_transform(X_train_test) 
X_test_test= Ro_scaler.transform(X_test_test)
```

### Creation of the Function for the Models/Algorithms
```python
def Evaluate(Model_Name, Model_Abb, X_test, Y_test):
    
    Pred_Value= Model_Abb.predict(X_test)
    Accuracy = metrics.accuracy_score(Y_test,Pred_Value)                      
    Sensitivity = metrics.recall_score(Y_test,Pred_Value)
    Precision = metrics.precision_score(Y_test,Pred_Value)
    F1_score = metrics.f1_score(Y_test,Pred_Value)
    Recall = metrics.recall_score(Y_test,Pred_Value)
    
    print('--------------------------------------------------\n')
    print('The {} Model Accuracy   = {}\n'.format(Model_Name, np.round(Accuracy,3)))
    print('The {} Model Sensitvity = {}\n'.format(Model_Name, np.round(Sensitivity,3)))
    print('The {} Model Precision  = {}\n'.format(Model_Name, np.round(Precision,3)))
    print('The {} Model F1 Score   = {}\n'.format(Model_Name, np.round(F1_score,3)))
    print('The {} Model Recall     = {}\n'.format(Model_Name, np.round(Recall,3)))
    print('--------------------------------------------------\n')
    
    Confusion_Matrix = metrics.confusion_matrix(Y_test, Pred_Value)
    plot_confusion_matrix(Confusion_Matrix,class_names=['Normal', 'Attack'],figsize=(5.55,5), colorbar= "blue")
    #plot_roc_curve(Model_Abb, X_test, Y_test)

def GridSearch(Model_Abb, Parameters, X_train, Y_train):
    Grid = GridSearchCV(estimator=Model_Abb, param_grid= Parameters, cv = 3, n_jobs=-1)
    Grid_Result = Grid.fit(X_train, Y_train)
    Model_Name = Grid_Result.best_estimator_
    
    return (Model_Name)
```

### 1: Logistic Regression Model
```python
### Logistic Regression
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
LR.fit(X_train_train , Y_train_train)

LR.score(X_train_train, Y_train_train), LR.score(X_test_train, Y_test_train)

Evaluate('Logistic Regression', LR, X_test_train, Y_test_train)
```
![LG_PERFORM](https://github.com/user-attachments/assets/95106df2-4916-4e36-bd3e-781bf148559b)
![LG_CFM](https://github.com/user-attachments/assets/432edbba-6011-4af4-abdd-5e3afccc75d8)

ROC Plot
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Your code for training the logistic regression model

# Get predicted probabilities for positive class
Y_prob = LR.predict_proba(X_test_train)[:, 1]

# Compute false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(Y_test_train, Y_prob)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
![LG_ROC](https://github.com/user-attachments/assets/0cc97459-3d82-4043-bcc4-1b2933d42869)


### 2: Decision Tree Model
```python
### Decision Tree
DT =DecisionTreeClassifier(max_features=6, max_depth=4)
DT.fit(X_train_train, Y_train_train)

DT.score(X_train_train, Y_train_train), DT.score(X_test_train, Y_test_train)

Evaluate('Decision Tree Classifier', DT, X_test_train, Y_test_train)
```
![DT_PERFORM](https://github.com/user-attachments/assets/37fe78f5-87e6-4de3-acb2-ce0cfae1e875)
![DT_CFM](https://github.com/user-attachments/assets/221dca1e-04db-4333-8cd4-c529254f86fb)

ROC Plot
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Your code for training the decision tree classifier model

# Get predicted probabilities for positive class
Y_prob = DT.predict_proba(X_test_train)[:, 1]

# Compute false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(Y_test_train, Y_prob)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
![DT_ROC](https://github.com/user-attachments/assets/36d78119-336e-4541-abdb-c62f8ee35c8a)

Plottin the Tree
```python
fig = plt.figure(figsize=(15,12))
tree.plot_tree(DT, filled=True)
```

![Tree](https://github.com/user-attachments/assets/1aaa92f3-83e5-48bc-bcc1-0227cab3374f)


### 3: Random Forest
```python
### Random Forest Classifier
max_depth= [1, 2, 3, 4, 5, 6, 7, 8, 9]
Parameters={ 'max_depth': max_depth}

RF= RandomForestClassifier()
GridSearch(RF, Parameters, X_train_train, Y_train_train)

RF.fit(X_train_train, Y_train_train)
RF.score(X_train_train, Y_train_train), RF.score(X_test_train, Y_test_train)

Evaluate('Random Forest Classifier', RF, X_test_train, Y_test_train)
```
![Random_PERFORM](https://github.com/user-attachments/assets/6b7816e9-ab26-4bdf-9ac7-7a9bbb1813a8)

![Random_CFM](https://github.com/user-attachments/assets/37b13b69-f9d4-4bc0-9a17-57b7335a5650)

ROC Plot
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the trained Random Forest Classifier model (RF) and the test data (X_test, Y_test)
y_pred_prob = RF.predict_proba(X_test_train)[:, 1]  # Predicted probabilities for the positive class
fpr, tpr, thresholds = roc_curve(Y_test_train, y_pred_prob)
auc = roc_auc_score(Y_test_train, y_pred_prob)

# Plotting the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # Plotting the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```
![Random_ROC](https://github.com/user-attachments/assets/839e74b7-4526-4efc-be92-5853c9ae7497)


### 4: KNN Model
```python
#### KNN Model
KNN= KNeighborsClassifier(n_neighbors=6) 
KNN.fit(X_train_train, Y_train_train)

KNN.score(X_train_train, Y_train_train), KNN.score(X_test_train, Y_test_train)
Evaluate('KNN', KNN, X_test_train, Y_test_train)
```
![KNN_PERFORM](https://github.com/user-attachments/assets/5061b27b-9530-4c9a-87cf-654f8b61b223)

![KNN_CFM](https://github.com/user-attachments/assets/c11cdd95-c22c-48d4-aef8-41ce9ebb0ac5)

ROC Plot
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the trained KNN model (KNN) and the test data (X_test_train, Y_test_train)
y_pred_prob = KNN.predict_proba(X_test_train)[:, 1]  # Predicted probabilities for the positive class
fpr, tpr, thresholds = roc_curve(Y_test_train, y_pred_prob)
auc = roc_auc_score(Y_test_train, y_pred_prob)

# Plotting the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # Plotting the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN')
plt.legend(loc="lower right")
plt.show()
```

![KNN_ROC](https://github.com/user-attachments/assets/f2518ded-a6fa-418b-acb7-232939ba1367)


### 5: SVM Model
```python
#### SVM Classifier
Linear_SVC = svm.LinearSVC(C=1)
Linear_SVC.fit(X_train_train, Y_train_train)

Linear_SVC.score(X_train_train, Y_train_train), Linear_SVC.score(X_test_train, Y_test_train)
Evaluate('SVM Linear SVC Kernel', Linear_SVC, X_test_train, Y_test_train)
```

![SVM_PERFORM](https://github.com/user-attachments/assets/336f4071-efc1-4bfe-bb2f-54414357330a)

![SVM_CFM](https://github.com/user-attachments/assets/be667bf2-7f84-4b02-a864-49c97d340280)

ROC Plot
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the trained SVM Linear SVC Kernel model (Linear_SVC) and the test data (X_test_train, Y_test_train)
y_pred_prob = Linear_SVC.decision_function(X_test_train)  # Predicted decision scores
fpr, tpr, thresholds = roc_curve(Y_test_train, y_pred_prob)
auc = roc_auc_score(Y_test_train, y_pred_prob)

# Plotting the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # Plotting the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM Linear SVC Kernel')
plt.legend(loc="lower right")
plt.show()
```

![SVM_ROC](https://github.com/user-attachments/assets/4c0fb3c8-f8d2-41b7-982a-f929dd9ae2b1)


### Predicting Stage
Uploading the data to be predicted for its class
```python
Pred_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Test_data_pred.csv')
Pred_Data
```
![Predicted_Data](https://github.com/user-attachments/assets/d9ac49b5-7613-46ad-8dba-fc5dc31a5e7b)

```python
# Predicting the class (anomaly or attack) using the trained model
predictions = Linear_SVC.predict(Pred_Data)

# Convert the predicted values back to their original labels
predicted_classes = attack_LE.inverse_transform(predictions)
# Attach predicted_classes as a column to Pred_Data
Pred_Data['predicted_class'] = predicted_classes

Pred_Data
```
![Last_Pred](https://github.com/user-attachments/assets/a086a0ec-230b-4eb2-bd0e-a508445eec80)



