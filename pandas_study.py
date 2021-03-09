
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('../data/solar/train/train.csv')
submission = pd.read_csv('../data/solar/sample_submission.csv')

df = pd.DataFrame(train)

tmp1 = df.copy()
tmp2 = df.copy()
tmp3 = df.copy()



# 판다스 삭제



# 인덱스 삭제 
tmp1 = tmp1.drop(tmp1.index[[0,1]])

# 컬럼의 필터 추가 
tmp1 = tmp1[tmp1.Minute >= 30]

# 컬럼의 드랍 (axis = 1 은 컬럼을 의미함)
tmp1 = tmp1.drop("Hour", axis=1)





# 판다스 컬럼 생성



tmp2['test'] = 0

# 컬럼 값에 조건으로 넣기 
tmp2['test'] = np.where(df['Minute'] != 30, 'yes', 'no')

# 사칙 연산도 가능 
tmp2['total'] = tmp2['Hour'] + tmp2["Minute"]

tmp2['total1'] = tmp2['Hour'] /2

# 판다스 컬럼으로 for문 가능 
grades = []
for i in df['Hour']:
    if i >= 12:
        grades.append('A')
    else:
        grades.append('B')
        

tmp2['grade'] = grades


# apply로 메소드 만들고 바로 적용시키기 가능함 
def test1(row):
    return "Pass"

tmp2['grade'] = tmp2.grade.apply(test1)

print(tmp2)




# 행 합치기

tmp3.append(tmp2, ignore_index=True)




# 판다스 그룹


student_list = [{'name': 'John', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Nate', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Abraham', 'major': "Physics", 'sex': "male"},
                {'name': 'Brian', 'major': "Psychology", 'sex': "male"},
                {'name': 'Janny', 'major': "Economics", 'sex': "female"},
                {'name': 'Yuna', 'major': "Economics", 'sex': "female"},
                {'name': 'Jeniffer', 'major': "Computer Science", 'sex': "female"},
                {'name': 'Edward', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Zara', 'major': "Psychology", 'sex': "female"},
                {'name': 'Wendy', 'major': "Economics", 'sex': "female"},
                {'name': 'Sera', 'major': "Psychology", 'sex': "female"}
         ]
df = pd.DataFrame(student_list, columns = ['name', 'major', 'sex'])

groupby_major = df.groupby('major')

for name, group in groupby_major:
    print(name + ": " + str(len(group)))
    print(group)
    print()


df_major_cnt = pd.DataFrame({'count' : groupby_major.size()}).reset_index()

print(df_major_cnt)


groupby_sex = df.groupby('sex')


for name, group in groupby_sex:
    print(name + ": " + str(len(group)))
    print(group)
    print()





# 판다스 중복 확인 

student_list = [{'name': 'John', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Nate', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Abraham', 'major': "Physics", 'sex': "male"},
                {'name': 'Brian', 'major': "Psychology", 'sex': "male"},
                {'name': 'Janny', 'major': "Economics", 'sex': "female"},
                {'name': 'Yuna', 'major': "Economics", 'sex': "female"},
                {'name': 'Jeniffer', 'major': "Computer Science", 'sex': "female"},
                {'name': 'Edward', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Zara', 'major': "Psychology", 'sex': "female"},
                {'name': 'Wendy', 'major': "Economics", 'sex': "female"},
                {'name': 'Sera', 'major': "Psychology", 'sex': "female"},
                {'name': 'John', 'major': "Computer Science", 'sex': "male"},
         ]
df = pd.DataFrame(student_list, columns = ['name', 'major', 'sex'])

print(df.duplicated())

df = df.drop_duplicates()

print(df)

df.duplicated(['name'])

#  name을 기준으로 중복값 찾는다 keep='first', keep='last' 앞 뒤 값으로 찾는다 
df.drop_duplicates(['name'], keep='last')







# nan 처리 

school_id_list = [{'name': 'John', 'job': "teacher", 'age': 40},
                {'name': 'Nate', 'job': "teacher", 'age': 35},
                {'name': 'Yuna', 'job': "teacher", 'age': 37},
                {'name': 'Abraham', 'job': "student", 'age': 10},
                {'name': 'Brian', 'job': "student", 'age': 12},
                {'name': 'Janny', 'job': "student", 'age': 11},
                {'name': 'Nate', 'job': "teacher", 'age': None},
                {'name': 'John', 'job': "student", 'age': None}
         ]
df = pd.DataFrame(school_id_list, columns = ['name', 'job', 'age'])

# nan 값 확인 

print(df.info())

print(df.isna())

print(df.isnull())


# null에 값 입력하기 

df.age = df.age.fillna(0)

print(df)

# 직업을 기준으로 나이컬럼에서 중간값을 넣어준다 
df["age"].fillna(df.groupby("job")["age"].transform("median"), inplace=True)







# apply 함수 


date_list = [{'yyyy-mm-dd': '2000-06-27'},
         {'yyyy-mm-dd': '2002-09-24'},
         {'yyyy-mm-dd': '2005-12-20'}]
df = pd.DataFrame(date_list, columns = ['yyyy-mm-dd'])

def extract_year(row):
    return row.split('-')[0]

df['year'] = df['yyyy-mm-dd'].apply(extract_year)



def extract_year(year, current_year):
    return current_year - int(year)

# apply에 변수 한개 받기 
df['age'] = df['year'].apply(extract_year, current_year=2018)



def get_introduce(age, prefix, suffix):
    return prefix + str(age) + suffix

# 다중 변수도 가능 
df['introduce'] = df['age'].apply(get_introduce, prefix="I am ", suffix=" years old")



def get_introduce2(row):
    return "I was born in "+str(row.year)+" my age is "+str(row.age)

# 판다스 로우에서 값 꺼내서 사용도 가능 
df.introduce = df.apply(get_introduce2, axis=1)





# map, applymap 사용 법


date_list = [{'yyyy-mm-dd': '2000-06-27'},
         {'yyyy-mm-dd': '2002-09-24'},
         {'yyyy-mm-dd': '2005-12-20'}]
df = pd.DataFrame(date_list, columns = ['yyyy-mm-dd'])


def extract_year(row):
    return row.split('-')[0]
# map 함수로 값 넣을 수 있다
df['year'] = df['yyyy-mm-dd'].map(extract_year)


job_list = [{'age': 20, 'job': 'student'},
         {'age': 30, 'job': 'developer'},
         {'age': 30, 'job': 'teacher'}]
df = pd.DataFrame(job_list)
# map 함수로 벨류를 변경할 수 있다 
df.job = df.job.map({"student":1,"developer":2,"teacher":3})


x_y = [{'x': 5.5, 'y': -5.6},
         {'x': -5.2, 'y': 5.5},
         {'x': -1.6, 'y': -4.5}]
df = pd.DataFrame(x_y)

# 데이터 전체를 바꾸고 싶으면 applymap을 사용하면 된다 
df = df.applymap(np.around)



# unique 사용법 


job_list = [{'name': 'John', 'job': "teacher"},
                {'name': 'Nate', 'job': "teacher"},
                {'name': 'Fred', 'job': "teacher"},
                {'name': 'Abraham', 'job': "student"},
                {'name': 'Brian', 'job': "student"},
                {'name': 'Janny', 'job': "developer"},
                {'name': 'Nate', 'job': "teacher"},
                {'name': 'Obrian', 'job': "dentist"},
                {'name': 'Yuna', 'job': "teacher"},
                {'name': 'Rob', 'job': "lawyer"},
                {'name': 'Brian', 'job': "student"},
                {'name': 'Matt', 'job': "student"},
                {'name': 'Wendy', 'job': "banker"},
                {'name': 'Edward', 'job': "teacher"},
                {'name': 'Ian', 'job': "teacher"},
                {'name': 'Chris', 'job': "banker"},
                {'name': 'Philip', 'job': "lawyer"},
                {'name': 'Janny', 'job': "basketball player"},
                {'name': 'Gwen', 'job': "teacher"},
                {'name': 'Jessy', 'job': "student"}
         ]
df = pd.DataFrame(job_list, columns = ['name', 'job'])



# 원하는 컬럼의 중복 제거하고 유니크한 값 나오도록 한다
print( df.job.unique() )

# 유니크한 벨류별로 몇개가 있는지 확인할 수 있는 코드 
df.job.value_counts()











# 판다스 합치기!

l1 = [{'name': 'John', 'job': "teacher"},
      {'name': 'Nate', 'job': "student"},
      {'name': 'Fred', 'job': "developer"}]

l2 = [{'name': 'Ed', 'job': "dentist"},
      {'name': 'Jack', 'job': "farmer"},
      {'name': 'Ted', 'job': "designer"}]
         
df1 = pd.DataFrame(l1, columns = ['name', 'job'])
df2 = pd.DataFrame(l2, columns = ['name', 'job'])


# 행으로 (아래로) 합치는 방법 ( ignore_index = True 해주면 인덱스를 다시 잡아준다 )
frames = [df1, df2]
result = pd.concat(frames, ignore_index=True)

l1 = [{'name': 'John', 'job': "teacher"},
      {'name': 'Nate', 'job': "student"},
      {'name': 'Fred', 'job': "developer"}]

l2 = [{'name': 'Ed', 'job': "dentist"},
      {'name': 'Jack', 'job': "farmer"},
      {'name': 'Ted', 'job': "designer"}]
         
df1 = pd.DataFrame(l1, columns = ['name', 'job'])
df2 = pd.DataFrame(l2, columns = ['name', 'job'])

result = df1.append(df2, ignore_index=True)


# 열로 데이터 합치기 

l1 = [{'name': 'John', 'job': "teacher"},
      {'name': 'Nate', 'job': "student"},
      {'name': 'Jack', 'job': "developer"}]

l2 = [{'age': 25, 'country': "U.S"},
      {'age': 30, 'country': "U.K"},
      {'age': 45, 'country': "Korea"}]
         
df1 = pd.DataFrame(l1, columns = ['name', 'job'])
df2 = pd.DataFrame(l2, columns = ['age', 'country'])
result = pd.concat([df1, df2], axis=1, ignore_index=True)
# axis = 1을 넣어주면 열로 합칠 수 있다 



# 배열 2개로 데이터 프레임 만들기 
label = [1,2,3,4,5]
prediction = [1,2,2,5,5]

comparison = pd.DataFrame(
    {'label': label,
     'prediction': prediction
    })




# 판다스 

# shfit(-1)  컬럼의 값을 위로 당긴다
# fillna : 데이터 당긴 nan 데이터를 원하는 데이터로 채운다 

