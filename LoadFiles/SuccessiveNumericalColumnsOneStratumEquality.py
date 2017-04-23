import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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


def createEqualStrataData(dataset, labels):
    highData = dataset.iloc[np.where(labels.values=='high')[0],:]
    highLabels = labels.iloc[np.where(labels.values=='high')[0]]    
    
    
    midData = dataset.iloc[np.where(labels.values=='medium')[0],:]
    midLabels = labels.iloc[np.where(labels.values=='medium')[0]]
    
    lowData = dataset.iloc[np.where(labels.values=='low')[0],:]
    lowLabels = labels.iloc[np.where(labels.values=='low')[0]]
    
    numDat = len(highData)

    midSampData, midSampLab = midData.sample(numDat), midLabels.sample(numDat)
    lowSampData, lowSampLab = lowData.sample(numDat), lowLabels.sample(numDat)
    
    newData = pd.concat([highData, midSampData, lowSampData])
    newLabels = pd.concat([highLabels,midSampLab,lowSampLab])

    return newData, newLabels

TSDat, TSLabs = createEqualStrataData(train_train_data,train_train_labels)

def editData(dataset):
    dataset['NumberFeats'] = [len(x) for x in dataset['features'].values]
    dataset['NumberPhoto'] = [len(x) for x in dataset['photos'].values]

    lat = sorted(dataset['latitude'].values.tolist())
    latEx0 = [x for x in lat if x != 0]
    dataset['lat'] = (dataset['latitude']-min(latEx0))/(max(latEx0)-min(latEx0))
    lon = sorted(dataset['longitude'].values.tolist())
    lonEx0 = [x for x in lon if x != 0]
    dataset['lon'] = (dataset['longitude']-min(lonEx0))/(max(lonEx0)-min(lonEx0))
    return dataset

newTrain = editData(TSDat)
newDev = editData(dev_data)

numbCols = ['bathrooms','bedrooms','price','listing_id','NumberFeats','NumberPhoto','lat','lon']
for i in range(len(numbCols)):
    columns = numbCols[:(i+1)]
    lr = LogisticRegression(penalty = 'l2',multi_class='multinomial',solver='newton-cg',max_iter=200)
    lr.fit(newTrain[columns],TSLabs.values)
    probs = lr.predict_proba(newDev[columns])
    print "With variables of:"
    print columns
    print "The dev log-loss is {:.3}".format(log_loss(dev_labels,probs))
    print " "
