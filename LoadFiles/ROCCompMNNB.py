import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings
from sklearn.feature_extraction.text import *
warnings.simplefilter(action = "ignore", category = FutureWarning)
pd.options.mode.chained_assignment = None

train_file = pd.read_json('C:/Users/maxst/OneDrive/Documents/Berkeley MIDS/Spring 2017 207/Projects/Final/train.json')
test_file = pd.read_json('C:/Users/maxst/OneDrive/Documents/Berkeley MIDS/Spring 2017 207/Projects/Final/test.json')

train_file = train_file.dropna()

###make splits
numbRows = train_file.count()[0]
percentTrain = 0.8
trainNumbs = round(numbRows*percentTrain,0)
indices = random.sample(range(numbRows),int(trainNumbs))
devIndices = list(set(range(numbRows))-set(indices))

train_data, train_labels = train_file.drop('interest_level',1), train_file['interest_level']
train_train_data, train_train_labels = train_data.iloc[indices,], train_labels.iloc[indices]
dev_data, dev_labels = train_data.iloc[devIndices,], train_labels.iloc[devIndices]

features = train_train_data['features']
FeaturesString = [' '.join(x) for x in features]
Finalvec = CountVectorizer()
trainSparse = Finalvec.fit_transform(FeaturesString)
def editData2(dataset):
    dataset['NumberFeats'] = [len(x) for x in dataset['features'].values]
    dataset['NumberPhoto'] = [len(x) for x in dataset['photos'].values]
    dataset['CategoricalBuild'] = pd.Categorical.from_array(dataset['building_id']).codes
    
    newData = dataset[['bathrooms','bedrooms','price','listing_id','NumberFeats','NumberPhoto',]]
    newData['CategoricalBuild'] = pd.Categorical.from_array(dataset['building_id']).codes
    newData['CategoricalManager'] = pd.Categorical.from_array(dataset['manager_id']).codes
    newData['CategoricalStreet'] = pd.Categorical.from_array(dataset['display_address']).codes
    
    features = dataset['features'].values
    FeaturesString = [' '.join(x) for x in features]
    sparse = Finalvec.transform(FeaturesString)
    featureDense = pd.DataFrame(sparse.todense(),index=dataset.index, columns = Finalvec.vocabulary_)
    finalData = newData.merge(featureDense,left_index=True,right_index=True)
    return finalData

newTrain2 = editData2(train_train_data)
newDev2 = editData2(dev_data)

lr = LogisticRegression(penalty = 'l2',multi_class='multinomial',solver='newton-cg',max_iter=200)
lr.fit(newTrain2,train_train_labels.values)
probs = lr.predict_proba(newDev2)
devLL = log_loss(dev_labels,probs)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(dev_labels, probs[:,0],pos_label=np.unique(dev_labels)[0])
AUCVal=auc(fpr, tpr)

fpr2, tpr2, thresholds2 = roc_curve(dev_labels, probs[:,2],pos_label=np.unique(dev_labels)[2])
AUCVal2=auc(fpr2, tpr2)

fpr1, tpr1, thresholds1 = roc_curve(dev_labels, probs[:,1],pos_label=np.unique(dev_labels)[1])
AUCVal1=auc(fpr1, tpr1)

plt.plot(fpr,tpr, label=np.unique(dev_labels)[0])
plt.plot(fpr2,tpr2, label=np.unique(dev_labels)[2])
plt.plot(fpr1,tpr1, label=np.unique(dev_labels)[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC by Class - Logistic Regression')



