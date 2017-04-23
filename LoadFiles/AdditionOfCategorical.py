import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings
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

def editData1(dataset):
    dataset['NumberFeats'] = [len(x) for x in dataset['features'].values]
    dataset['NumberPhoto'] = [len(x) for x in dataset['photos'].values]
    dataset['CategoricalBuild'] = pd.Categorical.from_array(dataset['building_id']).codes
    
    newData = dataset[['bathrooms','bedrooms','price','listing_id','NumberFeats','NumberPhoto',]]
    newData['CategoricalBuild'] = pd.Categorical.from_array(dataset['building_id']).codes
    newData['CategoricalManager'] = pd.Categorical.from_array(dataset['manager_id']).codes
    newData['CategoricalStreet'] = pd.Categorical.from_array(dataset['display_address']).codes
    return newData

newTrain1 = editData1(train_train_data)
dev1 = editData1(dev_data)

Catcols = ['CategoricalBuild','CategoricalManager','CategoricalStreet']
Num = ['bathrooms','bedrooms','price','listing_id','NumberFeats','NumberPhoto']
for i in range(len(Catcols)):
    columns = Catcols[:(i+1)]
    lr = LogisticRegression(penalty = 'l2',multi_class='multinomial',solver='newton-cg',max_iter=200)
    lr.fit(newTrain1[Num+columns],train_train_labels.values)
    probs = lr.predict_proba(dev1[Num+columns])
    print "With the addition of categorical variables of:"
    print columns
    print "The dev log-loss is {:.3}".format(log_loss(dev_labels,probs))
    print " "