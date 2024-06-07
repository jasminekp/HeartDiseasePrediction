import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

datafile = f'heart.csv'
df = pd.read_csv(datafile)
df.columns

# Check the shape of the dataset
df.shape

labels = (df['condition'] > 0)
features = df.drop(['condition'], axis = 1)

print(f'Disease cases: {sum(labels == 1):8d}')
print(f'Healthy cases: {sum(labels == 0):8d}')

# Study the features
features.head(10)

# Get the descriptive statistics
features.describe()

bin_features = ['sex', 'fbs', 'exang']
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_features = ['cp', 'restecg', 'slope', 'ca', 'thal']

N = len(bin_features)
fig, axs = plt.subplots(1, N, figsize=(12, 3))

for n in range(N):
    sns.histplot(data=features, x=bin_features[n], ax=axs[n])
    axs[n].set_ylabel('')
plt.show()

N = len(cat_features)
fig, axs = plt.subplots(1, N, figsize=(12, 3))

for n in range(N):
    sns.histplot(data=features, x=cat_features[n], ax=axs[n])
    axs[n].set_ylabel('')
plt.show()

N = len(num_features)
fig, axs = plt.subplots(1, N, figsize=(12, 3))

for n in range(N):
    sns.histplot(data=features, x=num_features[n], ax=axs[n])
    axs[n].set_ylabel('')
plt.show()

bin_values = features[bin_features].values
bin_values[:5]

transformer = preprocessing.RobustScaler().fit(features[num_features])
num_values = transformer.transform(features[num_features])
num_values[:5]

encoder = preprocessing.OneHotEncoder().fit(features[cat_features])
cat_values = encoder.transform(features[cat_features]).toarray()
cat_values[:5]

bin_values.shape, num_values.shape, cat_values.shape

all_values = np.concatenate((bin_values, num_values, cat_values), axis = 1)
all_values.shape

all_values[:1]

preprocessed_features = pd.DataFrame(all_values)
preprocessed_features.describe().T