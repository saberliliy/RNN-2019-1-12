import random
import os

import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.optimizers import Adam

from data_processing import *

class  model:
    def __init__(self):
        self.learning_rate = 0.001
        self.max_len = 6
        self.batch_size = 512
        self.word2numF = None
        self.num2word = None
        self.words = None
        self.files_content = None
        self.train_len=None
        self.word2numF, self.num2word, self.words, self.files_content = data_processing()
        self.train()
    def data_generator(self):
        i = 0
        while 1:
            x = self.files_content[i: i + self.max_len]
            y = self.files_content[i + self.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.max_len, len(self.words)),
                dtype=np.bool
            )

            for t, char in enumerate(x):
                x_vec[0, t, self.word2numF(char)] = 1.0

            yield x_vec, y_vec
            i += 1
    def sample(self, preds, temperature=1.0):

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    def generate_sample_result(self,epoch, logs):

        start_index = random.randint(0, len(self.files_content) - self.max_len - 1)
        generated = ''
        sentence = self.files_content[start_index: start_index + self.max_len]
        generated += sentence
        for i in range(20):
            x_pred = np.zeros((1, self.max_len,len(self.words)))
            for t, char in enumerate(sentence[-5:]):
                x_pred[0, t, self.word2numF(char)] = 1.0
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, 1.0)
            next_char = self.num2word[next_index]
            generated += next_char
            sentence = sentence + next_char
        print(sentence)

    def build_model(self):

        input_tensor = Input(shape=(self.max_len, len(self.words)))
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.words), activation='softmax')(dropout)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train(self):
        '''训练模型'''
        number_of_epoch = len(self.files_content) // self.batch_size


        self.build_model()

        self.model.summary()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint("poetry.h5", save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )
if __name__ == '__main__':
    model()