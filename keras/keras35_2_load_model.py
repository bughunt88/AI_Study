from tensorflow.keras.models import load_model
model = load_model('../data/h5/save_keras35.h5')

model.summary() #35-1파일과 같은 모델 써머리가 나온다. 윗줄에서 불러와졌음

# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.