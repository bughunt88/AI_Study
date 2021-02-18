from marchine.car import drive
from marchine.tv import watch



drive()
watch()

#from marchine import car
#from marchine import tv

from marchine import car, tv



car.drive()
tv.watch()


print("===================test==================")
from machine.test.test_car import drive
from machine.test.test_tv import watch

drive()
watch()


from machine.test import test_car
from machine.test import test_tv

test_car.drive()
test_tv.watch()



from machine import test
from machine import test

test.test_car.drive()
test.test_tv.watch()

