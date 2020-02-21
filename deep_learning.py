import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import pandas as pd

df = pd.read_csv("data/Exoplanet.csv")
df = df.dropna(axis='columns', how='all')
df = df.dropna()
df.head()

df_final = df[['koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'koi_period', 'koi_time0bk','koi_impact','koi_duration','koi_depth','koi_prad',
            'koi_teq','koi_insol','koi_model_snr','koi_tce_plnt_num','koi_steff','koi_slogg',
             'koi_srad','ra','dec','koi_kepmag']]
df_final.head()

df_final.isnull().values.any()

X = df_final.drop("koi_disposition", axis=1)
y = df_final["koi_disposition"]
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y = label_encoder.transform(y_train)
encoded_y = label_encoder.transform(y_test)

for label, original_class in zip(encoded_y, y):
    print('Original Class: ' + str(original_class))
    print('Encoded Label: ' + str(label))
    print('-' * 12)

from tensorflow.keras.utils import to_categorical

y_train_categorical = to_categorical(encoded_y)
y_test_categorical = to_categorical(encoded_y)
y_train_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

number_inputs = 20
number_hidden_nodes = 10
model.add(Dense(units=number_hidden_nodes,
                activation='relu', input_dim=number_inputs))

number_classes = 3
model.add(Dense(units=number_classes, activation='softmax'))

model.summary()


model.compile(optimizer='adam',
              loss='poisson',
              metrics=['accuracy'])

model.fit(
    X_test_scaled,
    y_train_categorical,
    epochs=1000,
    shuffle=True,
    verbose=2
)

model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")

encoded_predictions = model.predict_classes(X_test_scaled[:10])
prediction_labels = label_encoder.inverse_transform(encoded_predictions)

print(f"Predicted classes: {prediction_labels}")
print(f"Actual Labels: {list(y_test[:10])}")

