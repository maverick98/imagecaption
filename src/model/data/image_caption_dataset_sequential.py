#
#Ideally both RNN and Transformer should use the same ImageCaptionDataset class.
#There is no time for Software Engineering here.
#Thus this extra class
#

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt

import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector
from keras.layers import Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
#from tensorflow.keras.layers.wrappers import Bidirectional
#from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
#from tf.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from numpy import array
import pandas as pd
import cv2
from glob import glob
import PIL
import time
from tqdm import tqdm
import os
import gensim
import random
import time
     

class ImageCaptionDataset:
    def __init__(self,datasetInput, **kwargs):
        super().__init__(**kwargs)
        self.datasetInput=datasetInput
        self.tokens=None
        self.vocabulary=None
        self.all_images=None
        self.train_descriptions=None
        self.train_descriptions_raw=None
        self.all_train_captions=None

        self.dev_descriptions=None
        self.dev_descriptions_raw=None
        self.all_dev_captions=None

        self.test_descriptions=None
        self.test_descriptions_raw=None
        self.all_test_captions=None

        self.ixtoword=None
        self.wordtoix=None
        self.vocab_size=None
        self.embeddings_index=None
        self.embedding_matrix=None
        self.rnn_model=None
        self.descriptions_max_len=None
        self.encoded_train_images_features=None
        self.encoded_dev_images_features=None
        self.encoded_test_images_features=None
        self.model_loaded=False
    def show_random_images(self,how_many):
        all_images = glob(self.datasetInput.image_path + "*.jpg")
        rnd_images=random.choices(range(0,len(all_images)),k=how_many)
        for i in rnd_images:
            plt.figure()
            image = cv2.imread(all_images[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
    def load_descriptions(self):
        print("loading descriptions")
        file = open(self.datasetInput.token_file,'r')
        data = file.read()
        file.close()
        self.descriptions = dict()
        for line in data.split('\n'):
            col = line.split('\t')
            if len(col) == 1:
                continue
            image_id = col[0].split("#")[0].split(".")[0]
            image_desc= col[1]
            if image_id in self.descriptions:
                self.descriptions[image_id].append(image_desc)
            else:
                self.descriptions[image_id] = list()
                self.descriptions[image_id].append(image_desc)
        print("Found descriptions of legnth ",len(self.descriptions))          
        return  self.descriptions
    def clean_descriptions(self):
        print("Sanitizing descriptions")
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in self.descriptions.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                # tokenize
                desc = desc.split()
                # convert to lower case
                desc = [word.lower() for word in desc]
                # remove punctuation from each token
                desc = [w.translate(table) for w in desc]
                # remove hanging 's' and 'a'
                desc = [word for word in desc if len(word)>1]
                # remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # store as string
                desc_list[i] =  ' '.join(desc)                 
    
    # load clean descriptions into memory
    def load_clean_descriptions(self,dataset):
        # load document
        file_content = self.load_file(self.datasetInput.descriptions_file)
        #print(file_content)
        descriptions = dict()
        descriptions_raw = dict()
        for line in file_content.split('\n'):
            #print(line)
            # split line by white space
            tokens = line.split()
            # split id from description
            image_id, image_desc = tokens[0], tokens[1:]
            #print('image_id',image_id)
            #print('image_desc',image_desc)
            # skip images not in the set
            #print('image_data_path',self.get_image_path(image_id,self.image_path))
            if self.get_image_path(image_id,self.datasetInput.image_path) in dataset:
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                    descriptions_raw[image_id] = list()
                # wrap description in tokens
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                # store
                descriptions[image_id].append(desc)
                descriptions_raw[image_id].append(' '.join(image_desc))
        return descriptions,descriptions_raw
    def load_clean_train_descriptions(self):
        self.train_descriptions,self.train_descriptions_raw = self.load_clean_descriptions(self.train_imgs)
    def load_clean_dev_descriptions(self):
        self.dev_descriptions,self.dev_descriptions_raw = self.load_clean_descriptions(self.dev_imgs) 
    def load_clean_test_descriptions(self):
        self.test_descriptions,self.test_descriptions_raw = self.load_clean_descriptions(self.test_imgs)         
    # convert the loaded descriptions into a vocabulary of words
    def build_vocabulary(self):
        print("Building Vocabulary")
        # build a list of all description strings
        vocabulary = set()
        for key in self.descriptions.keys():
            [vocabulary.update(d.split()) for d in self.descriptions[key]]
        self.vocabulary=vocabulary
        print('Original Vocabulary Size: %d' % len(self.vocabulary))    
        return vocabulary
    def load_all_training_captions(self):
        # Create a list of all the training captions
        all_train_captions = []
        if self.train_descriptions == None:
            self.load_clean_train_descriptions()

        for key, val in self.train_descriptions.items():
            for cap in val:
                all_train_captions.append(cap)
        self.all_train_captions=all_train_captions
        return all_train_captions
    def load_all_dev_captions(self):
        # Create a list of all the training captions
        all_dev_captions = []
        if self.dev_descriptions == None:
            self.load_clean_dev_descriptions()

        for key, val in self.dev_descriptions.items():
            for cap in val:
                all_dev_captions.append(cap)
        self.all_dev_captions=all_dev_captions
        return all_dev_captions
    def load_all_test_captions(self):
        # Create a list of all the training captions
        all_test_captions = []
        if self.test_descriptions == None:
            self.load_clean_test_descriptions()

        for key, val in self.test_descriptions.items():
            for cap in val:
                all_test_captions.append(cap)
        self.all_test_captions=all_test_captions
        return all_test_captions
    def build_high_frequency_vocabularies(self):
        # Consider only words which occur at least 10 times in the corpus
        word_count_threshold = self.datasetInput.high_frequency_threshold
        word_counts = {}
        nsents = 0
        if self.all_train_captions == None:
            self.load_all_training_captions()
        if self.all_dev_captions == None:
            self.load_all_dev_captions()
        if self.all_test_captions == None:
            self.load_all_test_captions()    
        all_captions=[]
        for caption in self.all_train_captions:
            all_captions.append(caption)
        for caption in self.all_dev_captions:
            all_captions.append(caption)
        for caption in self.all_test_captions:
            all_captions.append(caption)                 

        for sent in all_captions:
            nsents += 1
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        self.ixtoword=ixtoword
        self.wordtoix=wordtoix
        vocab_size = len(ixtoword) + 1 # one for appended 0's
        self.vocab_size=vocab_size         
        self.high_frequency_train_vocabulary=vocab      
    # save descriptions to file, one per line
    def save_descriptions(self):
        print("Saving descriptions to ",self.datasetInput.descriptions_file)
        lines = list()
        for key, desc_list in self.descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(self.datasetInput.descriptions_file, 'w')
        file.write(data)
        file.close()
    
    # convert a dictionary of clean descriptions to a list of descriptions
    def convert_descriptions_to_lines(self):
        all_desc = list()
        for key in self.descriptions.keys():
            [all_desc.append(d) for d in self.descriptions[key]]
        self.descriptions_lines=all_desc    
        return self.descriptions_lines

    # calculate the length of the description with the most words
    def calculate_descriptions_max_len(self):
        lines = self.convert_descriptions_to_lines()
        self.descriptions_max_len=max(len(d.split()) for d in lines)
        return self.descriptions_max_len

    
    
    # load doc into memory
    def load_file(self,filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text
    def get_image_path(self,img, image_dataset_path):
        return image_dataset_path+img+'.jpg'      

    # load a pre-defined list of photo identifiers
    def load_set(self,filename):
        doc = self.load_file(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
            # skip empty lines
            if len(line) < 1:
                continue
            # get the image identifier
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)
    # Create a list of all image names in the directory
    def load_all_images(self):
        self.all_images = glob(self.datasetInput.image_path + '*.jpg')
        return self.all_images
    

    def process_tokens(self):
        print("Processing tokens")
        self.load_descriptions()
        self.clean_descriptions()
        self.build_vocabulary()
        self.save_descriptions()
    # Below file conatains the names of images to be used in train data  
    def load_train_test_imgs(self,train_test_images_file):
        if self.all_images == None:
            self.load_all_images()
        # Read the train image names in a set
        train_test_images = set(open(train_test_images_file, 'r').read().strip().split('\n'))
        #print(train_test_images)
        # Create a list of all the training images with their full path names
        train_test_img = []
        
        for i in self.all_images: # img is list of full path names of all images
            #train_test_img.append(i)
            if i[len(self.datasetInput.image_path):] in train_test_images: # Check if the image belongs to training set
                train_test_img.append(i) # Add it to the list of train images
        return train_test_img
    def load_train_imgs(self):
        self.train_imgs=self.load_train_test_imgs(self.datasetInput.train_images_text_file)
        print('Train imgs Size: %d' % len(self.train_imgs))    
    def load_dev_imgs(self):
        self.dev_imgs=self.load_train_test_imgs(self.datasetInput.dev_images_text_file)
        print('Dev imgs Size: %d' % len(self.dev_imgs))    
    def load_test_imgs(self):
        self.test_imgs=self.load_train_test_imgs(self.datasetInput.test_images_text_file) 
        print('Test imgs Size: %d' % len(self.test_imgs))
    def encode_train_images(self):
        self.encoded_train_images=self.datasetInput.cnn_model.encode_images(self.train_imgs)
    def encode_dev_images(self):
        self.encoded_dev_images=self.datasetInput.cnn_model.encode_images(self.dev_imgs)
    def encode_test_images(self):
        self.encoded_test_images=self.datasetInput.cnn_model.encode_images(self.test_imgs)
    def save_encoded_train_images(self):
        self.datasetInput.cnn_model.save_encoded_imgs(self.datasetInput.encoded_train_images_pkl,self.encoded_train_images)     
    def save_encoded_dev_images(self):
        self.datasetInput.cnn_model.save_encoded_imgs(self.datasetInput.encoded_dev_images_pkl,self.encoded_dev_images)    
    def save_encoded_test_images(self):
        self.datasetInput.cnn_model.save_encoded_imgs(self.datasetInput.encoded_test_images_pkl,self.encoded_test_images)
    def load_encoded_train_images(self):
        encoded_train_images_features = load(open(self.datasetInput.encoded_train_images_pkl, "rb"))
        print('Photos: train=%d' % len(encoded_train_images_features))
        self.encoded_train_images_features=encoded_train_images_features
        return self.encoded_train_images_features
    def load_encoded_dev_images(self):
        encoded_dev_images_features = load(open(self.datasetInput.encoded_dev_images_pkl, "rb"))
        print('Photos: dev=%d' % len(encoded_dev_images_features))
        self.encoded_dev_images_features=encoded_dev_images_features
        return self.encoded_dev_images_features
    def load_encoded_test_images(self):
        encoded_test_images_features = load(open(self.datasetInput.encoded_test_images_pkl, "rb"))
        print('Photos: test=%d' % len(encoded_test_images_features))
        self.encoded_test_images_features=encoded_test_images_features
        return self.encoded_test_images_features   

        
    def load_cnn_model(self):
        self.datasetInput.cnn_model.load_model()
    def show_cnn_model(self):
        self.datasetInput.cnn_model.show_summary()

    def load_word_embeddings(self):
        # Load Glove vectors
        
        embeddings_index = {} # empty dictionary
        f = open(os.path.join(self.datasetInput.glove_dir, self.datasetInput.glove_embedding_file), encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        self.embeddings_index=embeddings_index
    def build_word_embedding_matrix(self):                                       
        if self.embeddings_index == None:
            self.load_word_embeddings()

        # Get 200-dim dense vector for each of the 10000 words in out vocabulary
        embedding_matrix = np.zeros((self.vocab_size, self.datasetInput.embedding_dim))

        for word, i in self.wordtoix.items():
            #if i < max_words:
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix=embedding_matrix
    
    def build_rnn_model(self,mode='TRAIN'):
        if self.embedding_matrix == None:
            self.build_word_embedding_matrix()
        if self.descriptions_max_len ==None:
            self.calculate_descriptions_max_len()    
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(self.descriptions_max_len,))
        se1 = Embedding(self.vocab_size, self.datasetInput.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = tf.keras.layers.Add()([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        rnn_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        rnn_model.layers[2].set_weights([self.embedding_matrix])
        rnn_model.layers[2].trainable = False

        rnn_model.compile(loss='categorical_crossentropy', optimizer='adam')
        if mode == 'TRAIN':
            rnn_model.summary()
        
        self.rnn_model=rnn_model
    def save_model(self):
        self.rnn_model.save(self.datasetInput.model_weights_path+'/model_' + str(self.datasetInput.epochs) + '.h5')      
    def train_model(self):
        if self.rnn_model == None:
            self.build_rnn_model(mode='TRAIN')
        if self.encoded_train_images_features ==None:    
            self.load_encoded_train_images()  
        if self.encoded_dev_images_features ==None:    
            self.load_encoded_dev_images()      

        steps = len(self.train_descriptions)//self.datasetInput.num_photos_per_batch
        #for i in range(self.datasetInput.epochs):
        generator_train = self.data_generator(self.train_descriptions,self.encoded_train_images_features)
        #generator_val = self.data_generator(self.dev_descriptions,self.encoded_dev_images_features)
        #self.rnn_model.fit_generator(generator_train, epochs=self.datasetInput.epochs, steps_per_epoch=steps, verbose=1,validation_data =generator_val)
        self.rnn_model.fit_generator(generator_train, epochs=self.datasetInput.epochs, steps_per_epoch=steps, verbose=1)
        self.save_model()

    def load_model_weights(self):
        if self.model_loaded == False:
            if self.rnn_model == None:
                self.build_rnn_model(mode='TEST')
            self.rnn_model.load_weights(self.datasetInput.model_weights_path+'/model_' + str(self.datasetInput.epochs) + '.h5')     
            self.model_loaded=True
    def greedySearch(self,photo):
        self.load_model_weights()
        in_text = 'startseq'
        for i in range(self.descriptions_max_len):
            sequence = [self.wordtoix[w] for w in in_text.split() if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.descriptions_max_len)
            yhat = self.rnn_model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final
    def predict(self,z):
        if self.rnn_model == None:
            self.build_rnn_model(mode='TEST')
        encoding_test=self.load_encoded_test_images()
        pic = list(encoding_test.keys())[z]
        image = encoding_test[pic].reshape((1,2048))
        x=plt.imread(self.datasetInput.image_path+pic)
        plt.imshow(x)
        plt.show()
        print("Greedy:",self.greedySearch(image))
    

    # data generator, intended to be used in a call to model.fit_generator()  
    def data_generator(self,descriptions_dict,encoded_images_features):
        X1, X2, y = list(), list(), list()
        n=0
    
        while 1:
            for key, desc_list in descriptions_dict.items():
                n+=1
                # retrieve the photo feature
                photo = encoded_images_features[key+'.jpg']
                for desc in desc_list:
                    # encode the sequence
                    seq = [self.wordtoix[word] for word in desc.split(' ') if word in self.wordtoix]
                    # split one sequence into multiple X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=self.descriptions_max_len)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                        # store
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)
                # yield the batch data
                if n==self.datasetInput.num_photos_per_batch:
                    yield [[array(X1), array(X2)], array(y)]
                    X1, X2, y = list(), list(), list()
                    n=0


    def pretrain(self):
        print("Performing pretrain activities")
        self.process_tokens()
        self.load_train_imgs()
        self.load_dev_imgs()
        self.load_test_imgs()
        train_descriptions = self.load_clean_descriptions(self.train_imgs)
        print('Descriptions: train=%d' % len(train_descriptions))
        self.load_cnn_model()
        self.build_high_frequency_vocabularies()
               
     
     
                                                                                 