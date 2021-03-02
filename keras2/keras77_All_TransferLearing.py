from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

list_1 = [VGG16, Xception, VGG19, ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, 
DenseNet201, NASNetLarge, NASNetMobile, EfficientNetB0, EfficientNetB1]


for list in list_1:
        
    model = list()

    model.trainable = False

    model.summary()
