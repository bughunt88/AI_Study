import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(dataset.values())

print(dataset.target_names) # 타겟(y) 값의 원래 값
# ['setosa' 'versicolor' 'virginica']


# x = dataset.data
x = dataset['data']
# y = dataset.target
y = dataset['target']

print(x) # (150, 4
print(y)
print(x.shape, y.shape)
print(type(x), type(y))

# 타입 : <class 'numpy.ndarray'>

# df = pd.DataFrame(x, columns=dataset.feature_names)
df = pd.DataFrame(x, columns=dataset['feature_names'])
# 헤더는 데이터가 아니다 

print(df)
print(df.shape)
print(df.columns)
print(df.index)

# 일반적인 리스트는 shape가 먹히지 않는다

print(df.head())
# 위에서 부터 5개 보여주기 df[:5]

print(df.tail())
# 아래서 부터 5개 보여주기 df[-5:]

print(df.info())
# 정보를 보여주는 코드
# non-null은 null이 없다 비어있는 데이터가 없다

print(df.describe())
# 평균, min, max 값 정보를 보여준다 

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# 컬럼 명 변경

print(df.columns)
print(df.info())
print(df.describe())



# y컬럼을 추가해 보자
print(df['sepal_length'])

df['Target'] = dataset.target
# 새로운 컬럼과 데이터 넣는 방법 

print(df.shape) # (150,5)
print(df.columns)
print(df.index)
print(df.tail())

print(df.info())

print(df.isnull())
# null 값 확인
print(df.isnull().sum())
# 각 컬럼의 null 값 수 확인
print(df.describe())


print(df['Target'].value_counts())
# y값의 데이터가 같은 것들이 몇개씩 있는지 보여주는 코드 



# ***** 중요 *****


# 상관계수
print(df.corr())
# 타겟(y)에 계수가 1에 근접하면 관계가 좋은거다 
# 100프로 맞는건 아니지만 수치가 높으면 참고할 만하다
# 리니어로 볼 수 있다 



# 상관 계수 시각화!
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()



# 도수 분포도 
plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x='petal_length', data=df)
plt.title('petal_legnth')

plt.subplot(2,2,4)
plt.hist(x='petal_width', data=df)
plt.title('petal_width')



plt.show()

