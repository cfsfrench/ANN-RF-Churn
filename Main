import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow import keras as ks


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 40)

# Open customer data
df = pd.read_csv('Churn_Modeling.csv')

# Drop rows irrelevant to churn
df.drop('RowNumber', axis='columns', inplace=True)
df.drop('Surname', axis='columns', inplace=True)
df.drop('CustomerId', axis='columns', inplace=True)


# Include additional relationships based on values given
df['TenureByAge'] = df.Tenure/df.Age
df['CreditScoreGivenAge'] = df.CreditScore/df.Age
df['BalanceSalaryRatio'] = df.Balance/df.EstimatedSalary


# Transform all character values to numbers that can be read by the model
df.loc[df.HasCrCard == 0, 'HasCrCard'] = -1
df.loc[df.IsActiveMember == 0, 'IsActiveMember'] = -1

# Split Countries into separate columns with numbers that can be read by the model
lst = ['Geography', 'Gender']
remove = []
for i in lst:
    if df[i].dtype == np.str or df[i].dtype == np.object:
        for j in df[i].unique():
            df[i + '_' + j] = np.where(df[i] == j, 1, -1)
        remove.append(i)
df = df.drop(remove, axis=1)

# Preprocess data by fit and transform
scale = MinMaxScaler()
df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = \
    scale.fit_transform(df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']])

# Pop desired y value to new DF
y = df.pop('Exited')
X = df

# Split and randomize data with 80 - 20 ratio
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=200)

under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(x_train, y_train)

# Set RF parameters and fit the data
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0, warm_start=False)
RF.fit(X_res, y_res)
y_pred = RF.predict(x_test)

# Show classification report to assess accuracy compared to the ANN model
print("Random Forest Classifier report \n", classification_report(y_test, y_pred))

yp = []
for i in y_pred:
    if i > .5:
        yp.append(1)
    else:
        yp.append(0)

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
plt.figure(figsize=(10, 7))
sb.heatmap(cm, annot=True,  fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Now using ANN
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model = ks.Sequential([
    ks.layers.Dense(14, input_shape=(16,), activation='relu'),
    ks.layers.Dense(14, activation='relu'),
    ks.layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer='nadam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

yp = model.predict(x_test)


y_pred = []

for i in yp:
    if i > .5:
        y_pred.append(1)
    else:
        y_pred.append(0)


print(classification_report(y_test, y_pred))
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
plt.figure(figsize=(10, 7))
sb.heatmap(cm, annot=True,  fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
