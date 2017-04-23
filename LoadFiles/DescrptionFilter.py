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

def editData3(dataset):
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
    
    highData = train_train_data.iloc[np.where(train_train_labels.values=='high')[0],:]    
    lowData = train_train_data.iloc[np.where(train_train_labels.values=='low')[0],:]
    
    highCV = CountVectorizer()
    highCV.fit_transform(highData['description'])


    lowCV = CountVectorizer()
    lowCV.fit_transform(lowData['description'])
    
    highVocab = highCV.vocabulary_
    lowVocab = lowCV.vocabulary_
    highOnlyVocab = dict([(x, highVocab[x]) for x in highVocab.keys() if x not in lowVocab.keys()])
    descVec = CountVectorizer(vocabulary=list(set(highOnlyVocab)))
    descSparse = descVec.transform(dataset['description'])
    
    descDense = pd.DataFrame(descSparse.todense(),index=dataset.index, columns = descVec.vocabulary_)
    finalData1 = finalData.merge(descDense,left_index=True,right_index=True)
    return finalData1

newTrain3 = editData3(train_train_data)
newDev3 = editData3(dev_data)
newTest3 = editData3(test_file)

lr = LogisticRegression(penalty = 'l2',multi_class='multinomial',solver='newton-cg',max_iter=200)
lr.fit(newTrain3,train_train_labels.values)
probs = lr.predict_proba(newDev3)
print 'The dev log loss with high-interest vocab is {:.3}'.format(log_loss(dev_labels,probs))



