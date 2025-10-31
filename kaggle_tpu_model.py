import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU detected and initialized.")
except (ValueError, Exception):
    strategy = tf.distribute.get_strategy()
    print("TPU not found, using CPU/GPU strategy instead.")

# Data loading (Kaggle format expects data in ../input/) 
df = pd.read_csv('../input/hybrid-feature-selection-with-xgboost/data/AQbench_dataset.csv')
var_df = pd.read_csv('../input/hybrid-feature-selection-with-xgboost/data/AQbench_variables.csv')
hypers = pd.read_csv('../input/hybrid-feature-selection-with-xgboost/data/hyperparameters.csv')

def sine_cosine_encode(values, period=None):
    values_array = np.array(values)
    if period is None:
        period = values_array.max()
    sin_values = np.sin(2 * np.pi * values_array / period)
    cos_values = np.cos(2 * np.pi * values_array / period)
    return sin_values, cos_values

def preprocess(df, var_df):
    from sklearn.preprocessing import LabelEncoder
    str_columns = df.select_dtypes(include=['object']).columns.tolist()
    str_columns = [col for col in str_columns if col != 'dataset']
    for col in str_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('NaN'))
    df['lonx'], df['lony'] = sine_cosine_encode(df['lon'], period=360)
    df = df.drop('lon', axis=1)
    return df

df = preprocess(df, var_df)
input_cols = var_df.loc[(var_df['input_target'] == 'input') & (var_df['column_name'] != 'lon'), 'column_name'].tolist()
input_cols = [col for col in input_cols if col != 'lon']
input_cols += ['lonx', 'lony']
target_cols = hypers['column_name'].tolist()

x_train = df[df['dataset'] == 'train'][input_cols]
y_train = df[df['dataset'] == 'train'][target_cols]
x_val = df[df['dataset'] == 'val'][input_cols]
y_val = df[df['dataset'] == 'val'][target_cols]
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

def make_model(hp_row, num_inputs):
    h = eval(hp_row['hidden layers'])
    activation = hp_row['activation']
    l2_lambda = float(hp_row['L2 lambda'])
    inputs = keras.Input(shape=(num_inputs,), name='inputs')
    x = inputs
    for units in h:
        x = keras.layers.Dense(units, activation=activation, kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    outputs = keras.layers.Dense(1, name='output')(x)
    model = keras.Model(inputs, outputs, name=f"model_{hp_row['column_name']}")
    return model

models = {}
history = {}
for idx, hp_row in hypers.iterrows():
    col = hp_row['column_name']
    with strategy.scope():
        model = make_model(hp_row, x_train_scaled.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(hp_row['learning rate'])),
                      loss=hp_row['loss'],
                      metrics=['mae'])
    history[col] = model.fit(
        x_train_scaled,
        y_train[col].values,
        epochs=int(hp_row['epochs']),
        batch_size=int(hp_row['batch size']),
        validation_data=(x_val_scaled, y_val[col].values),
        verbose=2
    )
    models[col] = model
    print(f"Finished training for {col}")
# Save example model and history if needed
models[target_cols[0]].save('example_o3_model.h5')
# Save all if you wish (uncomment)
# for col, m in models.items():
#     m.save(f"model_{col}.h5")

# Optionally: create Kaggle submission prediction here (not implemented)
