#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 01:20:43 2022

@author: ceren
"""

import h5py as h5
import numpy as np

file = h5.File("eee443_project_dataset_train.h5", "r")

testFile = h5.File("eee443_project_dataset_test.h5", "r")

train_cap = file["train_cap"]  # training captions
test_cap = testFile["test_caps"]  # test captions

train_imid = file["train_imid"]  # training indices
test_imid = testFile["test_imid"]  # test indices

train_ims = file["train_ims"]
test_ims = testFile["test_ims"]

train_url = file["train_url"]  # url of training images
test_url = testFile["test_url"]  # url of test images

word_code = file["word_code"]
word_code = np.array(file["word_code"])