class Person:
    def  __init__(self, name, age, address):
        # self 클래스 자기 자신 
        self.name = name
        self.age = age
        self.address = address

    def greeting(self):
        print("안녕하세요, 저는 {0}입니다.".format(self.name))

    