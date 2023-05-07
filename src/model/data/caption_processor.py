import string
import tensorflow as tf
from time import time
import os
import numpy as np



class CaptionProcessor:
    def __init__(self,num_words,caption_max_len,glove_dir,glove_embedding_file,target_vocab_size,d_model,**kwargs):
        super().__init__(**kwargs)
        self.num_words=num_words
        self.caption_max_len=caption_max_len
        self.glove_dir=glove_dir
        self.glove_embedding_file=glove_embedding_file
        self.target_vocab_size=target_vocab_size
        self.d_model=d_model
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.num_words,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        
    def calculate_descriptions_max_len(self):
        lines = self.convert_descriptions_to_lines()
        self.descriptions_max_len=max(len(d.split()) for d in lines)
        return self.descriptions_max_len    
    # To remove punctuations
    def remove_punctuation(self,text_original):
        #print("text_original:::",text_original)
        text_no_punctuation = text_original.translate(string.punctuation)
        return(text_no_punctuation)
    def remove_frequency_based_stop_words(self,text,stop_words):
        result = ""
        for word in text.split():
            if word not in stop_words:
                result += " " + word
        return(result)
    # To remove single characters
    def remove_single_character(self,text):
        text_len_more_than1 = ""
        for word in text.split():
            if len(word) > 1:
                text_len_more_than1 += " " + word
        return(text_len_more_than1)

    # To remove numeric values
    def remove_numeric(self,text):
        text_no_numeric = ""
        for word in text.split():
            isalpha = word.isalpha()
            if isalpha:
                text_no_numeric += " " + word
        return(text_no_numeric)
    
    def sanitize_caption(self,text_original):
        text = self.remove_punctuation(text_original)
        text = self.remove_single_character(text)
        frequency_based_stop_words=['the','be']
        #text = self.remove_frequency_based_stop_words(text,frequency_based_stop_words)
        text = self.remove_numeric(text)
        return(text)
    def _sequence_pad(self,captions):
        captions_seqs = self.tokenizer.texts_to_sequences(captions)
        tokenized_captions=tf.keras.preprocessing.sequence.pad_sequences(captions_seqs, padding='post', maxlen= self.caption_max_len+1)
        return tokenized_captions

    def tokenize_captions(self,captions1,captions2,captions3):
        start_time=time()
        all=set(captions1).union(set(captions2)).union(set(captions3))
        self.tokenizer.fit_on_texts(all)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        tokenized_captions1=self._sequence_pad(captions1)
        tokenized_captions2=self._sequence_pad(captions2)
        tokenized_captions3=self._sequence_pad(captions3)
        
        print("Total time taken for tokenize_captions: %.2fs" % (time() - start_time))    
        return tokenized_captions1,tokenized_captions2,tokenized_captions3


    def tokenize_captionsOld(self,captions):
        start_time=time()
        self.tokenizer.fit_on_texts(captions)
        captions_seqs = self.tokenizer.texts_to_sequences(captions)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        captions_seqs = self.tokenizer.texts_to_sequences(captions)
        tokenized_captions=tf.keras.preprocessing.sequence.pad_sequences(captions_seqs, padding='post', maxlen= self.caption_max_len+1)
        print("Total time taken for tokenize_captions: %.2fs" % (time() - start_time))    
        return tokenized_captions
    def load_word_embeddings(self):
        # Load Glove vectors
        embeddings_index = {} # empty dictionary
        f = open(os.path.join(self.glove_dir, self.glove_embedding_file), encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        self.embeddings_index=embeddings_index
        return embeddings_index
    def build_word_embedding_matrix(self):                                       
        if self.embeddings_index == None:
            self.load_word_embeddings()

        # Get 200-dim dense vector for each of the 10000 words in out vocabulary
        embedding_matrix = np.zeros((self.target_vocab_size, self.d_model))

        for word, i in self.tokenizer.word_index.items():
            #if i < max_words:
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix=embedding_matrix
        