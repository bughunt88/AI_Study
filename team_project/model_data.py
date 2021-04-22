
def build_model(num) :

    if num == 0:
      inputs = Input(shape = (6,), name='input')
      x = Dense(1024, activation='swish', name='hidden1')(inputs)
      x = Dropout(0.2)(x)
      x = Dense(256, activation='swish', name='hidden2')(x)
      x = Dropout(0.2)(x)
      x = Dense(64, activation='swish', name='hidden3')(x)
      x = Dense(16, activation='swish', name='hidden4')(x)
      outputs = Dense(1, name='outputs')(x)
      model = Model(inputs=inputs, outputs=outputs)
      model.compile(loss='mse', optimizer='adam', metrics='mae')

    elif num == 1:
      inputs = Input(shape = (6,), name='input')
      x = Dense(1024, activation='selu', name='hidden1')(inputs)
      x = Dropout(0.2)(x)
      x = Dense(256, activation='selu', name='hidden2')(x)
      x = Dropout(0.2)(x)
      x = Dense(64, activation='selu', name='hidden3')(x)
      x = Dense(16, activation='selu', name='hidden4')(x)
      outputs = Dense(1, name='outputs')(x)
      model = Model(inputs=inputs, outputs=outputs)
      model.compile(loss='mse', optimizer='adam', metrics='mae')

    elif num == 2:
          inputs = Input(shape = (6,), name='input')
      x = Dense(1024, activation='Mish', name='hidden1')(inputs)
      x = Dropout(0.2)(x)
      x = Dense(256, activation='Mish', name='hidden2')(x)
      x = Dropout(0.2)(x)
      x = Dense(64, activation='Mish', name='hidden3')(x)
      x = Dense(16, activation='Mish', name='hidden4')(x)
      outputs = Dense(1, name='outputs')(x)
      model = Model(inputs=inputs, outputs=outputs)
      model.compile(loss='mse', optimizer='adam', metrics='mae')

    return model

 