from tensorflow.keras.preprocessing.text import Tokenizer

text = "나는 진짜 진짜 멋있는 법을 진짜 마구 마구 먹었다."

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

x = token.texts_to_sequences([text])

print(x)

from tensorflow.keras.utils import to_categorical

word_size = len(token.word_index)

print(word_size)

x = to_categorical(x)

print(x)
print(x.shape)

