import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.optimizers import Adam
from data_processing import *
import random
class predict:
    def __init__(self):
        self.model=None
        self.word2numF = None
        self.num2word = None
        self.words = None
        self.files_content = None
        self.train_len = None
        self.model=load_model("poetry.h5")
        self.max_len=6
        self.word2numF, self.num2word, self.words, self.files_content = data_processing()

    def sample(self, preds, temperature=1.0):

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    def predict(self, text):
        load_model("poetry.h5")
        with open("poetry.txt", 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)
        seed = random_line[-(self.max_len+1):-1]
        res = ''
        for c in text:
            seed = seed[1:] + c
            for j in range(5):
                x_pred = np.zeros((1, self.max_len, len(self.words)))
                for t, char in enumerate(seed):
                    x_pred[0, t] = self.word2numF(char)
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 1.0)
                next_char = self.num2word[next_index]
                seed = seed[1:] + next_char
            res += seed
        return res
if __name__ == '__main__':
    model=predict()
    while 1:
        text = input("text:")
        sentence = model.predict(text)
        print(sentence)