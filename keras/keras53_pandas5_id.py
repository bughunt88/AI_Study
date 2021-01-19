import pandas as pd

df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[7,8,9,10]], columns=list('abcd'), index=('가','나','다'))

print(df)

#판다스는 행 우선

# 아래처럼 해도 같은 데이터 프레임을 쓴다 
df2 = df

df2['a'] = 100


print(df2)
print(df)

print(id(df), id(df2))


# 아래처럼 해야 새로운 데이터 프레임으로 나뉜다 
# 사칙연산을 해서 새로운 값을 만들면 데이터 프레임 나뉜다
df3 = df.copy()
df2['b'] = 333

print(df)
print(df2)
print(df3)
