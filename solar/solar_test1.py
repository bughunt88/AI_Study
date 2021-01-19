import numpy as np
import pandas as pd

df = pd.read_csv('../data/solar/train/train.csv')

df.drop(['Hour','Minute','Day'], axis =1, inplace = True)
# print(df.shape) # (52560, 7)

data = df.to_numpy()
data = data.reshape(1095,48,6)

def split_xy(data,timestep,ynum):
    x,y = [],[]
    for i in range(len(data)):
        x_end = i + timestep
        y_end = x_end + ynum
        if y_end > len(data):
            break
        x_tmp = data[i:x_end]
        y_tmp = data[x_end:y_end,:,-1]
        x.append(x_tmp)
        y.append(y_tmp)
    return(np.array(x),np.array(y))
x,y = split_xy(data,7,2)

# x.shape = (1087,7,48,6)
# y.shape = (1087,2,48)  

# x,y = split_xy(data,7,2)
# x.shape = (1087,7,48,6)
# y.shape = (1087,2,48,6)

from sklearn.model_selection import train_test_split as tts
# x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8, shuffle = True, random_state = 0)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU, Reshape

drop = 0.3
model = Sequential()
model.add(Conv2D(512,2,padding = 'same', input_shape = (7,48,6)))
model.add(LeakyReLU(alpha = 0.05))
model.add(MaxPooling2D(2))
model.add(Dropout(drop))
model.add(Conv2D(256,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Conv2D(128,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Dense(128))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(256))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(2*48))
model.add(LeakyReLU(alpha = 0.05))
model.add(Reshape((2,48)))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10, verbose = 1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# 모델 9번 돌리기 
d = []  
for l in range(9):
   # cp = ModelCheckpoint(filepath = '../data/solar/data/modelcheckpoint/dacon%d.hdf5'%l,monitor='val_loss', save_best_only=True)
    model.fit(x,y,epochs= 1000, validation_split=0.2, batch_size =8, callbacks = [es,lr])

    c = []
    for i in range(81):
        testx = pd.read_csv('../data/solar/test/%d.csv'%i)
        testx.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        testx = testx.to_numpy()  
        testx = testx.reshape(7,48,6)
        testx,null_y = split_xy(testx,7,0)
        y_pred = model.predict(testx)
        y_pred = y_pred.reshape(2,48)
        a = []
        for j in range(2):
            b = []
            for k in range(48):
                b.append(y_pred[j,k])
            a.append(b)   
        c.append(a)
    d.append(c)
    # c = np.array(c) # (81, 2, 48)
d = np.array(d)
# print(d.shape) (9, 81, 2, 48)

print(d)

'''

### 뻘짓!! 쉐이프 바꿔주는중~~~
e = []
for i in range(81):
    f = []
    for j in range(2):
        g = []
        for k in range(48):
            h = []
            for l in range(9):
                h.append(d[l,i,j,k])
            g.append(h)
        f.append(g)
    e.append(f)

e = np.array(e)
df_sub = pd.read_csv('./practice/dacon/data/sample_submission.csv', index_col = 0, header = 0)

# submit 파일에 데이터들 덮어 씌우기!!
for i in range(81):
    for j in range(2):
        for k in range(48):
            df = pd.DataFrame(e[i,j,k])
            for l in range(9):
                df_sub.iloc[[i*96+j*48+k],[l]] = df.quantile(q = ((l+1)/10.),axis = 0)[0]

df_sub.to_csv('./practice/dacon/data/submit.csv')

'''