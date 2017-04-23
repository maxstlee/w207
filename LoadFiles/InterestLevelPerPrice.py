import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_file = pd.read_json('C:/Users/maxst/OneDrive/Documents/Berkeley MIDS/Spring 2017 207/Projects/Final/train.json')

f, (ax1, ax2) = plt.subplots(1, 2)
sns.set(style="whitegrid", color_codes=True)
sns.stripplot(x="interest_level", y="price", data=train_file, jitter=True,ax=ax1)
ax1.set_title('All Prices')

sns.stripplot(x="interest_level", y="price", data=train_file.iloc[np.where(train_file['price']<20000)[0],:], jitter=True,ax=ax2)
ax2.set_title('Focused')
plt.show()