#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:57:18 2022

@author: ceren
"""


import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import time
import os
import math
import tensorflow as tf
#import pickle
from pickle import load
from nltk.translate import bleu
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.models import Model


def seperateSentence(word_list):  # function for seperating sentences
    semi_first = ""
    semi_second = ""
    sentences_len = 17
    for i, word_index in enumerate(word_list.split(), 1):
        if (i // (math.ceil(sentences_len / 2))) > 0:
            semi_second = semi_second + " " + word_list
        else:
            semi_first = semi_first + " " + word_list
    return semi_first, semi_second


def caption_Image_Sampler(sample, word_list, dict, image_caption_listTest, wait_time, test):  # function for caption-image sampling
    semi_first, semi_second = seperateSentence(word_list)
    im_index = sample[1:]
    im_index = str(im_index)

    fullpath = "finIm/im" + im_index + ".png"
    if test == 1:
        fullpath = "testIm/im" + im_index + ".png"
    print(fullpath)
    if (os.path.exists(fullpath)):
        print("exists")
    image = cv.imread(fullpath)

    height, width = image.shape[:2]
    window_name = 'image'
    if test == 1:
        printCapt(sample, image_caption_listTest)
    else:
        printCapt(sample, dict)

    cv.putText(image, semi_first, (2, round(height - 20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.putText(image, semi_second, (2, round(height - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.imshow(window_name, image)
    cv.waitKey(wait_time)


def printCapt(sampleData, dict):  # function for printing multiple captions
    captionNumber = len(dict[sampleData])
    for i in range(captionNumber):
        printSingleCapt(dict[sampleData][i], i)


def printSingleCapt(capt, ind):  # function for printing a single caption
    countindexWord = np.count_nonzero(capt)
    strRet = "Caption " + str(ind + 1) + ": "
    for i in range(countindexWord - 2):
        strRet = strRet + word_list[capt[i + 1]] + " "
    print(strRet)


def getCapt(key, ind, dict):  # function for getting captions
    strRet = ""
    capt = dict[key][ind]
    countindexWord = np.count_nonzero(capt)
    for i in range(countindexWord - 2):
        strRet += word_list[capt[i + 1]] + " "
    return strRet


def splitTrainTest(trCap, trIm_id, trPer):  # function for splitting training and validation set
    np.random.seed(5)
    ind = np.arange(len(trCap))
    np.random.shuffle(ind)
    train_index_end = round(len(trCap) * trPer / 100)
    trCap = trCap[ind]
    trIm_id = trIm_id[ind]
    validationIm_id = trIm_id[0:train_index_end]
    validationCap = trCap[0:train_index_end]
    trIm_idRet = trIm_id[train_index_end:len(trIm_id)]
    trCap_ret = trCap[train_index_end:len(trIm_id)]
    return trIm_idRet, trCap_ret, validationIm_id, validationCap


def dataGenerator_2(train_ind, train_feature, batch_size=32, max_word_number=17, vocabulary_size=1004):  # function for generating captions
    imgData = np.empty((batch_size * (max_word_number - 1), 2048))
    capData = np.empty((batch_size * (max_word_number - 1), max_word_number))
    out = np.empty((batch_size * (max_word_number - 1), vocabulary_size))
    np.random.shuffle(train_ind)
    for batch in range(batch_size):
        imData = train_feature['k' + str(int(train_ind[batch]))]
        sequence = random.choice(image_caption_list['k' + str(int(train_ind[batch]))])
        for i in range(1, max_word_number):
            inSeq = sequence[:i]
            outSeq = sequence[i]
            inSeq = pad_sequences([inSeq], maxlen=max_word_number)[0]
            outSeq = to_categorical([outSeq], num_classes=vocabulary_size)[0]
            imgData[batch * (max_word_number - 1) + (i - 1), :] = imData
            capData[batch * (max_word_number - 1) + (i - 1), :] = inSeq
            out[batch * (max_word_number - 1) + (i - 1), :] = outSeq
    return [[imgData, capData], out]


def dataGenerator_Validation(validation_ind, inds, valFeat, batch_size=32, max_word_number=17, vocabulary_size=1004):  # function for generating captions
    imgData = np.empty((batch_size * (max_word_number - 1), 2048))
    capData = np.empty((batch_size * (max_word_number - 1), max_word_number))
    out = np.empty((batch_size * (max_word_number - 1), vocabulary_size))
    if (validation_ind + batch_size > len(inds)):
        batch_size = len(inds) - validation_ind
    for batch in range(batch_size):
        imageData = valFeat['k' + str(int(inds[validation_ind + batch]))]
        sequence = random.choice(image_caption_list['k' + str(int(inds[batch + validation_ind]))])
        for i in range(1, max_word_number):
            inSeq = sequence[:i]
            outSeq = sequence[i]
            inSeq = pad_sequences([inSeq], maxlen=max_word_number)[0]
            outSeq = to_categorical([outSeq], num_classes=vocabulary_size)[0]
            imgData[batch * (max_word_number - 1) + (i - 1), :] = imageData
            capData[batch * (max_word_number - 1) + (i - 1), :] = inSeq
            out[batch * (max_word_number - 1) + (i - 1), :] = outSeq

    return [[imgData, capData], out]


def predictionMaker(feat, ind, max_word_number):  # function for making predictions
    res = ""
    imgData = np.empty((1, 2048))
    imgData[0, :] = feat['k' + str(int(ind))]
    sequence = random.choice(image_caption_list['k' + str(int(ind))])
    predictionSeq = np.zeros((1, 17))
    predictionSeq[:, 0] = sequence[0]
    for i in range(1, max_word_number):
        prePadSeq = predictionSeq
        input = [imgData, prePadSeq]
        yPred = model.predict_on_batch(input)
        yPred = np.argmax(yPred)
        indexWord = word_list[yPred]
        if indexWord == 'x_END_':
            break
        res = res + " " + indexWord
        predictionSeq[0, i] = yPred
    return res


def bleuScore_Test():  # function for calculating bleu score for the test set
    print("====BLEU SCORE====")
    blueinitial_time = time.time()
    blueScore1 = 0
    blueScore2 = 0
    blueScore3 = 0
    blueScore4 = 0
    a = 0
    up = len(test_im_ind[0:len(test_im_ind)])
    for p in range(up):
        result = predictionMaker(test_features, test_im_ind[p], 17)
        result = result.split()
        related_captions = image_caption_listTest['k' + str(int(test_im_ind[p]))]
        references = []
        a = a + 1
        print(a)
        for j in range(len(related_captions)):
            references.append(getCapt('k' + str(int(test_im_ind[p])), j, image_caption_listTest).split())
        b1 = bleu(references, result, weights=(1, 0, 0, 0))  # 1 gram
        b1 = 0 if b1 < math.exp(-10) else b1
        b2 = bleu(references, result, weights=(0, 1, 0, 0))  # 2 gram
        b2 = 0 if b2 < math.exp(-10) else b2
        b3 = bleu(references, result, weights=(0, 0, 1, 0))  # 3 gram
        b3 = 0 if b3 < math.exp(-10) else b3
        b4 = bleu(references, result, weights=(0, 0, 0, 1))  # 4 gram
        b4 = 0 if b4 < math.exp(-10) else b4
        blueScore1 += b1
        blueScore2 += b2
        blueScore3 += b3
        blueScore4 += b4
    print("Bleu-1: ", 100 * (blueScore1 / len(test_im_ind)))
    print("Bleu-2: ", 100 * (blueScore2 / len(test_im_ind)))
    print("Bleu-3: ", 100 * (blueScore3 / len(test_im_ind)))
    print("Bleu-4: ", 100 * (blueScore4 / len(test_im_ind)))
    print('Time taken for bleu score {} sec\n'.format(time.time() - blueinitial_time))

#####################################################

# Load the datasets and parse them
dataset_file_training = h5.File("eee443_project_dataset_train.h5", "r") 
dataset_file_test = h5.File("eee443_project_dataset_test.h5", "r") 
training_image_index = dataset_file_training["train_imid"] 
test_image_index = dataset_file_test["test_imid"]
training_caption = dataset_file_training["train_cap"] 
test_caption = dataset_file_test["test_caps"] 
word_code = np.asarray(dataset_file_training['word_code']) # dictionary for converting words to vocabulary indices
# word_index = np.asarray(np.asarray(dataset_file_training['word_code']).tolist())

# Resize, squeeze, prepare the lists and arrays
word_index = np.asarray(word_code.tolist())
word_index = np.squeeze(word_index.astype(int))
word_index_sorted = np.argsort(word_index)
word_index = np.array(word_index)[word_index_sorted]
word_list = np.asarray(word_code.dtype.names)
word_list = np.squeeze(np.reshape(word_list, (1, 1004)))
word_list = np.array(word_list)[word_index_sorted]
word_index_sorted_2 = np.argsort(training_image_index)
training_image_index = np.array(training_image_index)[word_index_sorted_2]
training_caption = np.array(training_caption)[word_index_sorted_2]
training_image_index -= 1
image_caption_list = {}

for i in range(len(training_image_index)):
    if (('k' + str(training_image_index[i])) in image_caption_list):
        image_caption_list['k' + str(training_image_index[i])].append(training_caption[i])
    else:
        image_caption_list['k' + str(training_image_index[i])] = list()
        image_caption_list['k' + str(training_image_index[i])].append(training_caption[i])
image_caption_listTest = {}

for i in range(len(test_image_index)):
    if (('k' + str(test_image_index[i])) in image_caption_listTest):
        image_caption_listTest['k' + str(test_image_index[i])].append(test_caption[i])
    else:
        image_caption_listTest['k' + str(test_image_index[i])] = list()
        image_caption_listTest['k' + str(test_image_index[i])].append(test_caption[i])

# Load pickle dictionaries and keys for training, test and validation sets
training_dictionary = load(open("training_dict_2.pkl", "rb"))
training_keys = load(open("training_keys_new.pkl", "rb"))
train_im_ind = np.zeros(len(training_keys))
for i in range(len(training_keys)):
    train_im_ind[i] = int(training_keys[i][1:len(training_keys[i])])
    print(i)
    
test_dictionary = load(open("test_set.pkl", "rb"))
test_keys = load(open("test_keys.pkl", "rb"))
test_im_ind = np.zeros(len(test_keys))
for i in range(len(test_keys)):
    test_im_ind[i] = int(test_keys[i][1:len(test_keys[i])])

validation_dictionary = load(open("validation_dict_2.pkl", "rb"))
validation_keys = load(open("validation_keys_new.pkl", "rb"))
val_im_ind = np.zeros(len(validation_keys))
for i in range(len(validation_keys)):
    val_im_ind[i] = int(validation_keys[i][1:len(validation_keys[i])])


# Load pickle features for training, test and validation sets
train_features = load(open("training_feature_new.pkl", "rb"))
test_features = load(open("test_features.pkl", "rb"))
validation_features = load(open("validation_feature_new.pkl", "rb"))

# Some parameters
vocabulary_size = 1004
embedding_dimension = 300
max_word_number = 17
batch_size = 256
epoch_number = 50
num_of_batches = len(train_im_ind) // batch_size
validation_num_of_batches = len(val_im_ind) // batch_size

# Construct the model
image_features = Input(shape=(2048,))

layer_1 = Dropout(0.5)(image_features)
layer_2 = Dense(256, activation='relu')(layer_1)

language_input = Input(shape=(max_word_number,))

sequential_layer = Embedding(vocabulary_size, embedding_dimension, mask_zero=True)(language_input)
sequential_layer_2 = Dropout(0.5)(sequential_layer)
sequential_layer_3 = LSTM(256)(sequential_layer_2)

decoding_layer_1 = tf.concat([layer_2, sequential_layer_3], axis=-1)
decoding_layer_2 = Dense(256, activation='relu')(decoding_layer_1)

output_layer = Dense(vocabulary_size, activation='softmax')(decoding_layer_2)

model = Model(inputs=[image_features, language_input], outputs=output_layer)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam')


validation_loss_store = []
training_loss_store = []

epoch_iteration = 1

# Iterate through epochs
for i in range(epoch_number):
    # Start training
    initial_time = time.time()
    epoch_loss = 0
    batch_iteration = 0
    
    # Iterate through batch
    for j in range(num_of_batches):
        batch_iteration = batch_iteration + 1
        [X, y] = dataGenerator_2(np.copy(train_im_ind), train_features, batch_size=batch_size)
        batch_loss = model.train_on_batch(X, y)
        epoch_loss = epoch_loss + batch_loss
        
    average_training_loss = epoch_loss / num_of_batches
    training_loss_store.append(average_training_loss)
    print('Current Epoch {} Training Loss {:.5f}'.format(epoch_iteration, average_training_loss))
    print('Elapsed time for training: {} sec\n'.format(time.time() - initial_time))
    
    # Start validation
    validation_initial_time = time.time()
    validation_epoch_loss = 0
    
    # Iterate through batch
    for k in range(validation_num_of_batches):  # +1 is required to make a pass on every example
        [X_val, y_val] = dataGenerator_Validation(k * batch_size, val_im_ind, validation_features, batch_size=batch_size)
        validation_batch_loss = model.test_on_batch(X_val, y_val)
        validation_epoch_loss += validation_batch_loss
        
    average_validation_loss = validation_epoch_loss / validation_num_of_batches
    validation_loss_store.append(average_validation_loss)
    print('Current Epoch {} Validation Loss {:.5f}'.format(epoch_iteration, average_validation_loss))
    print('Elapsed time for validation: {} sec\n'.format(time.time() - validation_initial_time))
    epoch_iteration = epoch_iteration + 1
    
    # Plot loss versus epoch curves for training and validation
    plt.figure(0)
    plt.plot(training_loss_store)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.figure(1)
    plt.plot(validation_loss_store)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs. Epoch')
    plt.show()

# Calculate Bleu score for test    
bleuScore_Test()

