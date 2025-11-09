from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Permute, RepeatVector, Flatten, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import ops
from utils import *
import numpy as np
import xgboost as xgb

def attention_3d_block(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # Use ops.shape for dynamic shape or hard-code based on known architecture
    time_steps = 20  # From training data shape (3478, 20, 7)
    input_dim = 7  # INPUT_DIMS
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: ops.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(INPUT_DIMS = 13,TIME_STEPS = 20,lstm_units = 64):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # padding = 'same'
    x = Dropout(0.3)(x)

    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def PredictWithData(data,data_yuan,name,modelname,INPUT_DIMS = 13,TIME_STEPS = 20):
    print(data.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _ = create_dataset(data)
    _, testY = create_dataset(data_y)
    print("testX Y shape is:", testX.shape, testY.shape)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)

    model = attention_model(INPUT_DIMS)
    model.load_weights(modelname)
    model.summary()
    y_hat =  model.predict(testX)
    testY, y_hat = xgb_scheduler(data_yuan, y_hat)
    return y_hat, testY

def lstm(model_type,X_train,yuan_X_train):
    if model_type == 1:
        # single-layer LSTM
        model = Sequential()
        model.add(LSTM(units=50, activation='relu',
                    input_shape=(X_train.shape[1], 1)))
        model.add(Dense(units=1))
        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu',
                    input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(Dense(units=5))
    if model_type == 2:
        # multi-layer LSTM
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(1))

        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(LSTM(units=50, activation='relu'))
        yuan_model.add(Dense(5))
    if model_type == 3:
        # BiLSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'),
                                input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))

        yuan_model = Sequential()
        yuan_model.add(Bidirectional(LSTM(50, activation='relu'),
                                    input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(Dense(5))

    return model,yuan_model

def xgb_scheduler(data,y_hat):
    close = data.pop('close')
    data.insert(5, 'close', close)
    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2

def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # print('train', train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # print('trainX', trainX, 'trainy', trainy)
    # fit model - reduced estimators and added max_depth to reduce memory usage
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def walk_forward_validation(train, test):
    predictions = list()
    train = train.values
    history = [x for x in train]
    # print('history', history)
    # Limit history size to prevent memory issues (keep last 500 samples)
    max_history_size = 500
    for i in range(len(test)):
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        # print('i', i, testX, testy)
        # Limit history to prevent memory overflow
        train_subset = history[-max_history_size:] if len(history) > max_history_size else history
        yhat = xgboost_forecast(train_subset, testX)
        predictions.append(yhat)
        history.append(test.iloc[i, :])
        print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
    return test.iloc[:, -1],predictions