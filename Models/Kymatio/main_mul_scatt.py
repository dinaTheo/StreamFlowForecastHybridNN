import time as pytime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.layers import  LSTM, Bidirectional, Dense, Input, Concatenate, Reshape, Lambda, DepthwiseConv2D, Conv2D, Activation, Permute
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from hydroeval import evaluator, nse, kge
from datetime import datetime
from kymatio import Scattering1D

folder_with_data = 'Data'
river_flow = pd.read_csv(folder_with_data + '/river-flow-2.csv')
river_flow['dateTime'] = pd.to_datetime(river_flow['dateTime'], errors='raise')
river_flow['dateTime'] = river_flow['dateTime'].dt.date

climate_variables = pd.read_csv(folder_with_data + '/atmospheric-variables-2.csv')
climate_variables['dateTime'] = pd.to_datetime(climate_variables['valid_time'], errors='raise')
climate_variables['dateTime'] = climate_variables['dateTime'].dt.date
climate_variables = climate_variables.drop(columns={'number', 'expver', 'valid_time'})
climate_variables = climate_variables.groupby(['longitude','latitude','dateTime']).mean().reset_index()
climate_variables = climate_variables.drop(columns={'longitude', 'latitude'})
weather_data = pd.merge(climate_variables, river_flow, on=['dateTime'], how='inner')
weather_data = weather_data.drop(columns={'Unnamed: 0'})
weather_data = weather_data.iloc[:,[0,1,3,4,5,7,6,8,9,10,11,12,13]]

time_steps = 30 
hours = 24
horizon = 7 

numerical_columns = weather_data.select_dtypes(include=['float64']).columns
datetime_columns = weather_data.select_dtypes(include=['datetime64']).columns

scaler = MinMaxScaler(feature_range=(0, 1))
weather_data[numerical_columns]= pd.DataFrame(scaler.fit_transform(weather_data[numerical_columns]), columns = numerical_columns)

Y =  weather_data.iloc[:, 7:]  
X1 = weather_data.iloc[:, [1]]  
X2 = weather_data.iloc[:, [2]] 
X3 = weather_data.iloc[:, [3]] 
X4 = weather_data.iloc[:, [4]] 
X5 = weather_data.iloc[:, [5]] 
X6 = weather_data.iloc[:, [6]] 
time = weather_data.iloc[:, [0]] 
# 
Y = np.array(Y)
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
X5 = np.array(X5)
X6 = np.array(X6)
Y = Y.reshape(-1, hours, Y.shape[1])
X1 = X1.reshape(-1, hours, X1.shape[1])
X2 = X2.reshape(-1, hours, X2.shape[1])
X3 = X3.reshape(-1, hours, X3.shape[1])
X4 = X4.reshape(-1, hours, X4.shape[1])
X5 = X5.reshape(-1, hours, X5.shape[1])
X6 = X6.reshape(-1, hours, X6.shape[1])

def create_sequences(X1, X2, X3, X4, X5, X6, 
                     Y, time, time_steps, horizon):
    
    X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq = [], [], [], [], [], []
    
    Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq = [], [], [], [], [], []
    
    Y_seq = []
    
    for i in range(len(Y) - time_steps - horizon + 1):
        X1_seq.append(X1[i:i+time_steps])
        X2_seq.append(X2[i:i+time_steps])
        X3_seq.append(X3[i:i+time_steps])
        X4_seq.append(X4[i:i+time_steps])
        X5_seq.append(X5[i:i+time_steps])
        X6_seq.append(X6[i:i+time_steps])

        Y1_past_seq.append(Y[i:i+time_steps, :, 0]) 
        Y2_past_seq.append(Y[i:i+time_steps, :, 1])  
        Y3_past_seq.append(Y[i:i+time_steps, :, 2]) 
        Y4_past_seq.append(Y[i:i+time_steps, :, 3])  
        Y5_past_seq.append(Y[i:i+time_steps, :, 4]) 
        Y6_past_seq.append(Y[i:i+time_steps, :, 5]) 

        Y_seq.append(Y[i+time_steps:i+time_steps+horizon, :, :])  

    X1_seq = np.array(X1_seq)
    X2_seq = np.array(X2_seq)
    X3_seq = np.array(X3_seq)
    X4_seq = np.array(X4_seq)
    X5_seq = np.array(X5_seq)
    X6_seq = np.array(X6_seq)
    
    Y1_past_seq = np.array(Y1_past_seq)
    Y2_past_seq = np.array(Y2_past_seq)
    Y3_past_seq = np.array(Y3_past_seq)
    Y4_past_seq = np.array(Y4_past_seq)
    Y5_past_seq = np.array(Y5_past_seq)
    Y6_past_seq = np.array(Y6_past_seq)
    
    Y_seq = np.array(Y_seq)
    
    return (X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, 
            Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq,
            Y_seq
            )

X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq, Y_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y, time, time_steps, horizon
)

modality_1 = Y1_past_seq
modality_2 = Y2_past_seq
modality_3 = Y3_past_seq
modality_4 = Y4_past_seq
modality_5 = Y5_past_seq
modality_6 = Y6_past_seq

X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, modality_1_train,modality_1_test,modality_2_train,modality_2_test,modality_3_train, modality_3_test,modality_4_train,modality_4_test,modality_5_train,modality_5_test,modality_6_train,modality_6_test,y_train, y_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, modality_1,modality_2,modality_3,modality_4,modality_5,modality_6, Y_seq, test_size=0.2, shuffle=False)

modality_1_train = modality_1_train.reshape(-1,time_steps, hours, 1)
modality_2_train = modality_2_train.reshape(-1,time_steps, hours, 1)
modality_3_train = modality_3_train.reshape(-1,time_steps, hours, 1)
modality_4_train = modality_4_train.reshape(-1,time_steps, hours, 1)
modality_5_train = modality_5_train.reshape(-1,time_steps, hours, 1)
modality_6_train = modality_6_train.reshape(-1,time_steps, hours, 1)
modality_1_test = modality_1_test.reshape(-1,time_steps, hours,1)
modality_2_test = modality_2_test.reshape(-1,time_steps, hours,1)
modality_3_test = modality_3_test.reshape(-1,time_steps, hours,1)
modality_4_test = modality_4_test.reshape(-1,time_steps, hours,1)
modality_5_test = modality_5_test.reshape(-1,time_steps, hours,1)
modality_6_test = modality_6_test.reshape(-1,time_steps, hours,1)

inputs = np.stack([
    X1_train,X2_train,X3_train,X4_train,X5_train,X6_train,
    modality_1_train,modality_2_train,modality_3_train,modality_4_train,modality_5_train, modality_6_train],axis=3)
test_inputs = np.stack([
    X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, 
    modality_1_test, modality_2_test, modality_3_test, modality_4_test, modality_5_test, modality_6_test], axis=3)

inputs = inputs.reshape(inputs.shape[0]*inputs.shape[1],inputs.shape[2],inputs.shape[3])
test_inputs = test_inputs.reshape(test_inputs.shape[0]*test_inputs.shape[1],test_inputs.shape[2],test_inputs.shape[3])
print(inputs.shape)
channels = []
test_channels = []

# loop per feature (12 features: past river flow per station & past climatic variables)
for i in range(12): 
    inputs_new = inputs[:,:,i]
    test_inputs_new = test_inputs[:,:,i]

    scattering = Scattering1D(J=4, shape=24, Q=8)
    scattered = scattering(inputs_new)
    scattered_test = scattering(test_inputs_new)

    channels.append(scattered)
    test_channels.append(scattered_test)

# need to check this output
scattered = np.stack(channels, axis=2)
scattered_test = np.stack(test_channels, axis=2)
print(scattered.shape)

scattered = scattered.transpose(0,3,1,2)
scattered_test = scattered_test.transpose(0,3,1,2)
scattered = scattered.reshape(-1,time_steps,scattered.shape[2],scattered.shape[3])
scattered = scattered_test.reshape(-1,time_steps,scattered_test.shape[2],scattered_test.shape[3])

inputs = scattered
test_inputs = scattered_test

# Model parameters
TIME_STEPS = inputs.shape[1]
N_SCATTER = inputs.shape[2]
# Features are 12 = modals
N_MODALS = inputs.shape[3]
HORIZON = 7 
# Features are 12 in reality, leave it to 1, as their being merged later
N_FEATURES = 1 
print("y_train",y_train.shape)
print("y_test",y_test.shape)
inputs = [inputs[:, :, :, i:i+1] for i in range(N_MODALS)]
test_inputs = [test_inputs[:, :, :, i:i+1]  for i in range(N_MODALS)] 
INPUTS = inputs 
# Features are 12 in reality, leave it to 1, as their being merged later
LSTM_UNITS = 64

# CNN-BiLSTM as per Hybrid Scattering LSTM paper
# CNN  
def cnn_module(input_shape):
    inputs = Input(shape=input_shape)
    x = DepthwiseConv2D(kernel_size=(input_shape[0], 1), padding='valid')(inputs)
    x = Conv2D(filters=N_SCATTER, kernel_size=1, padding='valid')(x)
    # x = BatchNormalization()(x)  Adding BatchNormalization per paper is not improving performance
    x = Activation('relu')(x)
    return Model(inputs, x) 
# BiLSTM
def lstm_module(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=False))(inputs)
    # x = BatchNormalization()(x)
    x = Dense(7 * 6)(x)  
    x = Reshape(7, 6)(x)
    return Model(inputs, x) 
# plain early fusion = no fusion, just concatenation of features
def fusion(modalities): 
    concatenated = Concatenate()(modalities)  
    return concatenated
# Complete model
def build_model(time_steps, n_features, n_scatter, n_modals, horizon, inputs):
    inputs = [Input(shape=(time_steps, n_scatter, n_features)) for _ in range(n_modals)]

    fusion_output = fusion(inputs) 
    print(fusion_output.shape)

    fusion_output = Permute((1, 3, 2))(fusion_output)  
    print(fusion_output.shape)

    fusion_output = Lambda(lambda t: tf.reshape(t, (-1, n_modals, n_scatter, 1)))(fusion_output)
    print(fusion_output.shape)

    cnn_outputs = cnn_module(input_shape=(n_modals, n_scatter, 1))(fusion_output)
    print(cnn_outputs.shape)
    cnn_outputs = Lambda(lambda t: tf.reshape(t, (-1, time_steps, n_scatter*n_scatter)))(cnn_outputs) 
    print(cnn_outputs.shape)
    lstm_outputs = lstm_module((time_steps, cnn_outputs.shape[2]))(cnn_outputs)
    model = Model(inputs=inputs, outputs=lstm_outputs)
    model.compile(optimizer='adam' , loss='mean_squared_error')
    return model
def ensemble_predict(models, x): 
    predictions = np.array([model.predict(x) for model in models])  
    avg_predictions = np.mean(predictions, axis=0)  
    return avg_predictions

n_runs = 10
durations = []
durations_predictions = []
models = []
predictionstosave = []

for i in range(n_runs):
    model = build_model(TIME_STEPS, N_FEATURES, N_SCATTER, N_MODALS, HORIZON, INPUTS)  
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    start_time = pytime.time()  
    history = model.fit(inputs, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    end_time = pytime.time() 
    duration = end_time - start_time

    durations.append(duration)
    models.append(model)

    # print(f"Training time: {duration:.2f} seconds")
    best_epoch = np.argmin(history.history['val_loss']) + 1
    # print(f"The best epoch is {best_epoch}")

    test_loss = model.evaluate(test_inputs, y_test)
    # print(f'Test Loss: {test_loss}')
    start_time2 = pytime.time()
    prediction = model.predict(test_inputs)
    end_time2  = pytime.time()
    duration2 = end_time2 - start_time2
    # print(prediction.shape)  
    # print(f"Prediction time: {duration2:.2f} seconds")
    durations_predictions.append(duration2)
    if i==5:
        model.save('Results/base-mul-kymatio.h5')
    predictionstosave.append(prediction)

np.save(f'Results/full-predictions-base-mul-kymatio.npy', predictionstosave)

scaler_modified = MinMaxScaler(feature_range=(0, 1))
scaler_modified.min_ = scaler.min_[6:]
scaler_modified.scale_ = scaler.scale_[6:]
scaler_modified.data_min_ = scaler.data_min_[6:]
scaler_modified.data_max_ = scaler.data_max_[6:]
scaler_modified.data_range_ = scaler.data_range_[6:]
scaler_modified.feature_range = scaler.feature_range

predictions = ensemble_predict(models, test_inputs)
np.save(f'Results/predictions-base-mul-kymatio.npy', predictions)

samples, horizon, stations = y_test.shape
y_test2 = y_test.reshape(-1, stations)
y_test_unscaled = scaler_modified.inverse_transform(y_test2)

rmse_per_run  = []
mse_per_run  = []
mae_per_run  = []
mae_high_per_run = []
mape_per_run = []
nse_per_run  = []
kge_per_run  = []

for preds in predictionstosave:
    samples, horizon, stations = preds.shape
    preds2 = preds.reshape(-1, stations)
    preds_unscaled = scaler_modified.inverse_transform(preds2)

    df_actual = pd.DataFrame(y_test_unscaled.flatten(), columns=['Flow'])
    df_pred = pd.DataFrame(preds_unscaled.flatten(), columns=['Flow'])   

    threshold_high_actual = df_actual['Flow'].quantile(0.95)
    threshold_high_pred = df_actual['Flow'].quantile(0.95)

    extreme_highs_actual = df_actual[df_actual['Flow'] >= threshold_high_actual]
    extreme_highs_pred   = df_pred[df_actual['Flow'] >= threshold_high_pred]

    rmse = mean_squared_error(y_test_unscaled.flatten(), preds_unscaled.flatten(), squared=False)
    mse = mean_squared_error(y_test_unscaled.flatten(), preds_unscaled.flatten(), squared=True)
    mae = mean_absolute_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    mae_high = mean_absolute_error(extreme_highs_actual, extreme_highs_pred)
    mape = mean_absolute_percentage_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    nse_value = evaluator(nse, np.array(preds_unscaled.flatten()), np.array(y_test_unscaled.flatten()))
    kge_value = evaluator(kge, np.array(preds_unscaled.flatten()), np.array(y_test_unscaled.flatten()))
    
    rmse_per_run.append(round(rmse, 2))
    mse_per_run.append(round(mse, 2))
    mae_per_run.append(round(mae, 2))
    mae_high_per_run.append(round(mae_high, 2))
    mape_per_run.append(round(mape, 2))
    nse_per_run.append(round(nse_value[0], 2))
    kge_per_run.append(round(kge_value[0,0], 2))

durations_and_metrics_df = pd.DataFrame({
  'Run': [f'{i+1}' for i in range(n_runs)],
    'duration': durations,
    'prediction-duration': durations_predictions,
    'RMSE': rmse_per_run,
    'MSE':  mse_per_run,
    'MAE':  mae_per_run,
    'MAE_HIGH':  mae_high_per_run,
    'MAPE': mape_per_run,
    'NSE':  nse_per_run,
    'KGE':  kge_per_run,            
})

durations_and_metrics_df.to_csv('Results/durations-and-metrics-kymatio.csv')

