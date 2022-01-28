# simple personal project
# consist of applying some machine/deep learning techniques to the concrete dataset

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import numpy as np
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv('/home/lucas04/Documents/Projets/learning/concrete.csv')
data.head()
features = data.drop('CompressiveStrength', 1)
features.head()
features.shape[1]
label = data.CompressiveStrength

features = data.copy()
# Remove target
label = features.pop('CompressiveStrength')


preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

features = preprocessor.fit_transform(features)
label = np.log(label)

features = pd.DataFrame(features)

input_shape = [features.shape[1]]
print("Input shape: {}".format(input_shape))

data.head()     # original data
features.head() # tran
train_feat, test_feat, train_label, test_label = train_test_split( features, data.CompressiveStrength,test_size=0.3, random_state=0)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer = "adam", loss = "mae")

history = model.fit(features, label, epochs = 100, batch_size = 128 )

history_df = pd.DataFrame(history.history)

history_df.loc[3:, ['loss']].plot()
plt.show()

history = model.fit(
    train_feat, train_label,
    validation_data=(test_feat, test_label),
    batch_size=128,
    epochs=50,
    verbose=0 
)

history_df = pd.DataFrame(history.history)
history_df.head()

history_df.loc[5:, ['loss', 'val_loss']].plot()
plt.show()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

# here it seems we have some bias
# we decide to add some capacity to our model

model_c = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),    
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])
    
history_c = model_c.fit(
     train_feat, train_label,
     validation_data=(test_feat, test_label),
     batch_size=256,
     epochs=100,
     verbose=0)


history_c_df = pd.DataFrame(history_c.history)
history_c_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()
print("Minimum Validation Loss: {:0.4f}".format(history_c_df['val_loss'].min()))

# we add a stopping time to stop the network in order to avoid overfitting
model_c = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=input_shape),
    layers.Dense(256, activation='relu'),    
    layers.Dense(256, activation='relu'),
    layers.Dense(1)
])

model_c.compile(optimizer='adam',loss='mae')

early_stopping = EarlyStopping(min_delta = 0.01, patience = 5, restore_best_weights = True)

history_c = model_c.fit(
    train_feat, train_label,
    validation_data=(test_feat, test_label),
    batch_size=128,
    callbacks=[early_stopping],
    epochs=50
)

history_c_df = pd.DataFrame(history_c.history)
history_c_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()


# here we will experiment dropout 
model_d = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=input_shape),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])


model_d.compile(optimizer='adam',loss='mae')

history_d = model_d.fit(
    train_feat, train_label,
    validation_data=(test_feat, test_label),
    batch_size=256,
    epochs=100,
    callbacks=[early_stopping],
    verbose=0)


# Show the learning curves
history_d_df = pd.DataFrame(history_d.history)
history_d_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()

# let's the same model but optimizer (stochastic gradient descent)
model_c = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])


model_c.compile(optimizer='sgd',loss='mae')

early_stopping = EarlyStopping(min_delta = 0.01, patience = 5, restore_best_weights = True)

history_c = model_c.fit(
    train_feat, train_label,
    validation_data=(test_feat, test_label),
    batch_size=64,
    epochs=100
)

history_c_df = pd.DataFrame(history_c.history)
history_c_df.loc[5:, ['loss', 'val_loss']].plot()
plt.show()







# example with binary classification (radar data from https://archive.ics.uci.edu/ml/datasets/Ionosphere )

from IPython.display import display

ion = pd.read_csv('/home/lucas04/Documents/Projets/learning/ion.csv', index_col=0)
display(ion.head())

df = ion.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True) 
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
y_train = df_train['Class']
y_valid = df_valid['Class']

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid')
])

# the last layer is activate with a sigmoid since we want a probability (between 0 and 1)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
# since we consider a binary classification, we need to use the appropriate loss and metrics

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0
)

history_df = pd.DataFrame(history.history)

history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))

