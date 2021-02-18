import p71_byunsu.py

print(p71.aaa)
print(p71.square(10))

print("========================")

from p71_byunsu.py import aaa, square

aaa = 3


print(aaa)

print(square(10))