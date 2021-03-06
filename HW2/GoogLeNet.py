'''
MLP character model. Code adapted from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Embedding

from keras import layers
from keras.layers import Input
from keras.models import Model

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


def build_model(maxlen, chars):
    print('Build model...')

    convInputShape = Input(shape=(maxlen, len(chars)))

    tower_0 = Conv1D(64, 1, padding='same', activation ='relu')(convInputShape)

    tower_1 = Conv1D(64, 1, padding='same', activation ='relu')(convInputShape)
    tower_1 = Conv1D(64, 3, padding='same', activation ='relu')(tower_1)

    tower_2 = Conv1D(64, 1, padding='same', activation ='relu')(convInputShape)
    tower_2 = Conv1D(64, 7, padding='same', activation ='relu')(tower_2)

    tower_3 = Conv1D(64, 1, padding='same', activation='relu')(convInputShape)
    tower_3 = Conv1D(64, 20, padding='same', activation='relu')(tower_3)

    concat = layers.concatenate([tower_0, tower_1, tower_2, tower_3])
    # print(concat.shape)

    poolingLayer = MaxPooling1D(pool_size=4, strides=None, padding='same')
    pooledValues = poolingLayer(concat)
    # print(poolingLayer.input_shape)
    # print(poolingLayer.output_shape)

    dropoutLayer = Dropout(0.5)
    dropoutResult = dropoutLayer(pooledValues)

    mlpInputLayer = Flatten()
    mlpInputLayerOutput = mlpInputLayer(dropoutResult)
    # print(mlpInputLayer.input_shape)
    # print(mlpInputLayer.output_shape)

    mlpHiddenLayer = Dense(128, activation='relu', name='1Dense')
    mlpHiddenLayerOutput = mlpHiddenLayer(mlpInputLayerOutput)

    mlpOutputLayer = Dense(len(chars), activation='softmax', name='2Dense')
    mlpOutputLayerOutput = mlpOutputLayer(mlpHiddenLayerOutput)

    model = Model(inputs=convInputShape, outputs=mlpOutputLayerOutput)
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == '__main__':
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model = build_model(maxlen, chars)

    model.fit(
        x,
        y,
        batch_size=16,
        epochs=1,
        callbacks=[print_callback]
    )