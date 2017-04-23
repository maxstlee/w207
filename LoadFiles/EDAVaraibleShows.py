import pandas as pd

train_file = pd.read_json('C:/Users/maxst/OneDrive/Documents/Berkeley MIDS/Spring 2017 207/Projects/Final/train.json')
test_file = pd.read_json('C:/Users/maxst/OneDrive/Documents/Berkeley MIDS/Spring 2017 207/Projects/Final/test.json')

train_data, train_labels = train_file.drop('interest_level',1), train_file['interest_level']


