#https://github.com/maverick98/Group4Capstone/blob/main/ImageCaptionDataset_demo.ipynb
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import os
#import gensim
from sklearn.model_selection import train_test_split
from src.model.data.cnn_model import CNN_Model
from src.model.data.caption_processor import CaptionProcessor

class ImageCaptionDataset:
    def __init__(self,model_training_params,model_params, **kwargs):
        super().__init__(**kwargs)
        self.cnn_model=CNN_Model(cnn_type='Inception')
        image_path=model_training_params.image_path
        

        token_file=model_training_params.token_file

        token_file_train=model_training_params.token_file_train
        token_file_val=model_training_params.token_file_val
        token_file_test=model_training_params.token_file_test
        
        data_limit_train=model_training_params.data_limit_train
        data_limit_val=model_training_params.data_limit_val
        data_limit_test=model_training_params.data_limit_test
        
        num_words=model_training_params.num_words
        caption_max_len=model_training_params.caption_max_len
        batch_size=model_training_params.batch_size
        use_train_val_test_split=model_training_params.use_train_val_test_split
        glove_dir=model_training_params.glove_dir
        glove_embedding_file=model_training_params.glove_embedding_file
        target_vocab_size=model_params.target_vocab_size
        d_model=model_params.d_model

        self.use_train_val_test_split=use_train_val_test_split

        self.captionProcessor=CaptionProcessor(num_words,caption_max_len,glove_dir,glove_embedding_file,target_vocab_size,d_model)
        self.image_path=image_path
        self.token_file=token_file    
        self.token_file_train=token_file_train
        self.token_file_val=token_file_val
        self.token_file_test=token_file_test

        self.data_limit_train=data_limit_train
        self.data_limit_val=data_limit_val
        self.data_limit_test=data_limit_test
        
        self.batch_size=batch_size    
        
        self.image_caption_df_all=None
        self.image_caption_df_train=None #This dataframe holds the entire image-caption pairs
        self.image_caption_df_val=None #This dataframe holds the entire image-caption pairs
        self.image_caption_df_test=None #This dataframe holds the entire image-caption pairs
        
        
        self.all_img_names_train=None
        self.all_img_names_val=None
        self.all_img_names_test=None

        
        
        self.all_captions_train=None
        self.all_captions_val=None
        self.all_captions_test=None
        
        self.dataset_train=None
        self.dataset_val=None
        self.dataset_test=None


        self.clean_vocabulary_train=None
        self.clean_vocabulary_train_size=None
        
        self.clean_vocabulary_val=None
        self.clean_vocabulary_val_size=None

        self.clean_vocabulary_test=None
        self.clean_vocabulary_test_size=None
 
        
    
    #embedding_matrix=load_word_embeddings()
    def load_img_captions_train(self,img_file_name):
        return self._load_img_captions(img_file_name)
    def load_img_captions_val(self,img_file_name):
        return self._load_img_captions(img_file_name)
    def load_img_captions_test(self,img_file_name):
        return self._load_img_captions(img_file_name)

    #
    # This loads  captions for an given image.
    # It essentially loads all the five captions
    #
    def _load_img_captions(self,img_file_name):
        img_captions=[]
        
        for cap in self.image_caption_df_all[self.image_caption_df_all.filename==img_file_name].caption:
            img_captions.append([cap])
        return img_captions
        #return self.captionProcessor.tokenize_captions(img_captions)    
        

    def load_all_captions(self,image_caption_df):
        all_captions = []

        for caption  in image_caption_df["caption"].astype(str):
            #caption = '<start> ' + caption+ ' <end>'
            all_captions.append(caption)
        return all_captions    
    
    def load_all_image_names(self,image_caption_df):
        all_img_names = []
        for annot in image_caption_df["filename"]:
            full_image_path = self.image_path + annot
            all_img_names.append(full_image_path)
        return    all_img_names 

    def load_dataframe_train(self,image_caption_df):
        
        self.image_caption_df_train=image_caption_df
        #Load all the captions
        self.all_captions_train=self.load_all_captions(self.image_caption_df_train)      
        #Load all the image names
        self.all_img_names_train=self.load_all_image_names(self.image_caption_df_train)
        #Limit the dataset for debugging purpose
        self.all_captions_train,self.all_img_names_train=self.limit_dataset(self.all_captions_train,self.all_img_names_train,self.data_limit_train) 

    def load_dataframe_val(self,image_caption_df):
        
        self.image_caption_df_val=image_caption_df
        #Load all the captions
        self.all_captions_val=self.load_all_captions(self.image_caption_df_val)      
        #Load all the image names
        self.all_img_names_val=self.load_all_image_names(self.image_caption_df_val)
        #Limit the dataset for debugging purpose
        self.all_captions_val,self.all_img_names_val=self.limit_dataset(self.all_captions_val,self.all_img_names_val,self.data_limit_val) 
    def load_dataframe_test(self,image_caption_df):
        
        self.image_caption_df_test=image_caption_df
        #Load all the captions
        self.all_captions_test=self.load_all_captions(self.image_caption_df_test)      
        #Load all the image names
        self.all_img_names_test=self.load_all_image_names(self.image_caption_df_test)
        #Limit the dataset for debugging purpose
        self.all_captions_test,self.all_img_names_test=self.limit_dataset(self.all_captions_test,self.all_img_names_test,self.data_limit_test) 
    def load_dataframe(self):
        
        if self.use_train_val_test_split == True:
            image_caption_df = self._load_datafrme_with_captions(self.token_file_train)
            self.load_dataframe_train(image_caption_df)
            image_caption_df = self._load_datafrme_with_captions(self.token_file_val)
            self.load_dataframe_val(image_caption_df)
            image_caption_df = self._load_datafrme_with_captions(self.token_file_test)
            self.load_dataframe_test(image_caption_df)
        else:
            self._load_dataframe_all(self.token_file)
            #self.image_caption_df_all is already loaded      
            train, rest = train_test_split(self.image_caption_df_all, train_size=0.8)
          
            self.load_dataframe_train(train)
            val, test = train_test_split(rest, test_size=0.5)
      
            self.load_dataframe_val(val)
  
            self.load_dataframe_test(test)
        self.process_captions()
        self.all_captions_train,self.all_captions_val,self.all_captions_test=self.captionProcessor.tokenize_captions(self.all_captions_train,self.all_captions_val,self.all_captions_test)    

    def _load_datafrme_with_captions(self,token_file):
        self._load_dataframe_all(self.token_file)
        df = self._load_dataframe(token_file)
        df_final = pd.merge(df, self.image_caption_df_all, on='filename', how='left')
        df_final= df_final[df_final["index"].notna()]
        return df_final

    def _load_dataframe(self,token_file):
        file = open(token_file,'r')
        text = file.read()
        file.close()
        datatxt = []
        for line in text.split('\n'):
            datatxt.append(line)
        df = pd.DataFrame(datatxt,columns=["filename"])
        df = df.reindex(columns =['filename'])
        #One stupid image with extra .1 in it. Thus ignore it.
        df = df[df['filename'] != '2258277193_586949ec62.jpg.1']
        
        return df
    def _load_dataframe_all(self,token_file):
        if self.image_caption_df_all is not None:
            return
        print("Loading image and captions from  {}".format(token_file))
        file = open(token_file,'r')
        text = file.read()
        file.close()

        datatxt = []
        for line in text.split('\n'):
            col = line.split('\t')
            if len(col) == 1:
                continue
            w = col[0].split("#")
            datatxt.append(w + [col[1].lower()])

        image_caption_df = pd.DataFrame(datatxt,columns=["filename","index","caption"])
        image_caption_df = image_caption_df.reindex(columns =['index','filename','caption'])
        #One stupid image with extra .1 in it. Thus ignore it.
        image_caption_df = image_caption_df[image_caption_df['filename'] != '2258277193_586949ec62.jpg.1']
        image_caption_df= image_caption_df[image_caption_df["index"].notna()]
        self.image_caption_df_all=image_caption_df
        
    def show_stats(self):
        jpgs = os.listdir(self.image_path)
        print("Total Images in Dataset = {}".format(len(jpgs)))
        print("Showing stats for Train ")
        self._show_stats(self.image_caption_df_train)
        print("Showing stats for Val ")
        self._show_stats(self.image_caption_df_val)
        print("Showing stats for Test ")
        self._show_stats(self.image_caption_df_test)

    def _show_stats(self,image_caption_df):
        
        vocabulary = []
        stop_count=image_caption_df.shape[0]-1
        for i, txt in enumerate(image_caption_df.caption.values):
            #print("count is ",i)
            if i == stop_count:
                break
            vocabulary.extend(txt.split())
        print('Vocabulary Size: %d' % len(set(vocabulary)))
        img = image_caption_df["filename"].tolist()
        caption = image_caption_df["caption"].tolist()
        print(f"len(images) : {len(img)}")
        print(f"len(captions) : {len(caption)}")

 
    def process_captions(self):
        self.all_captions_train=self._sanitize_captions(self.all_captions_train)
        self.all_captions_train=self._surround_captions_with_special_token(self.all_captions_train)
        self.all_captions_train_size=len(set(self.all_captions_train))
        print('Clean Train Vocabulary Size: ', self.all_captions_train_size)
        
        self.all_captions_val=self._sanitize_captions(self.all_captions_val)
        self.all_captions_val=self._surround_captions_with_special_token(self.all_captions_val)
        self.all_captions_val_size=len(set(self.all_captions_val))
        print('Clean Train Vocabulary Size: ', self.all_captions_val)

        self.all_captions_test=self._sanitize_captions(self.all_captions_test)
        self.all_captions_test=self._surround_captions_with_special_token(self.all_captions_test)
        self.all_captions_test_size=len(set(self.all_captions_test))
        print('Clean Train Vocabulary Size: ', self.all_captions_test_size)
        
    def sanitize_captionsOld(self):
        self.clean_vocabulary_train=self._sanitize_captionsOld(self.image_caption_df_train)
        self.clean_vocabulary_train_size=len(set(self.clean_vocabulary_train))
        print('Clean Train Vocabulary Size: ', self.clean_vocabulary_train_size)
        
        self.clean_vocabulary_val=self._sanitize_captionsOld(self.image_caption_df_val)
        self.clean_vocabulary_val_size=len(set(self.clean_vocabulary_val))
        print('Clean Val Vocabulary Size: ', self.clean_vocabulary_val_size)
        
        self.clean_vocabulary_test=self._sanitize_captionsOld(self.image_caption_df_test)
        self.clean_vocabulary_test_size=len(set(self.clean_vocabulary_test))
        print('Clean Test Vocabulary Size: ', self.clean_vocabulary_test_size)
    def _sanitize_captions(self,image_captions):
        print("image_captions size ",len(image_captions))
        new_img_captions = [ self.captionProcessor.sanitize_caption(caption) for caption in image_captions]
        return new_img_captions
    def _surround_captions_with_special_token(self,image_captions):
        
        new_img_captions = [ '<start> ' + caption+ ' <end>' for caption in image_captions]
        return new_img_captions
    
    def _sanitize_captionsOld(self,image_caption_df):
        print("df size ",image_caption_df.shape)
        stop_count=image_caption_df.shape[0]-1
        clean_vocabulary = []
        for i, caption in enumerate(image_caption_df.caption.values):
            #print("count is ",i)
            if i == stop_count:
                break
            newcaption = self.captionProcessor.sanitize_caption(caption)
            image_caption_df["caption"].iloc[i] = newcaption
            clean_vocabulary.extend(newcaption.split())
       
        #for txt in image_caption_df.caption.values:
        #    clean_vocabulary.extend(txt.split())
        return clean_vocabulary
        #self.clean_vocabulary_size=len(set(clean_vocabulary))
        #print('Clean Vocabulary Size: ', self.clean_vocabulary_size)

    
    
         
    def limit_dataset(self,all_captions,all_img_names,data_limit):
        all_captions, all_img_names = shuffle(all_captions,all_img_names,random_state=1)
        all_captions = all_captions[:data_limit]
        all_img_names = all_img_names[:data_limit]
        return all_captions,all_img_names
    
    def tokenize_captions(self):
        self.all_captions_train=self._tokenize_captions(self.all_captions_train)
        self.all_captions_val=self._tokenize_captions(self.all_captions_val)
        self.all_captions_test=self._tokenize_captions(self.all_captions_test)
    def _tokenize_captions(self,all_captions):
        return self.captionProcessor.tokenize_captions(all_captions)
    
    def preprocess_images(self):
        self._preprocess_images(self.all_img_names_train)
        self._preprocess_images(self.all_img_names_val)
        self._preprocess_images(self.all_img_names_test)
    def _preprocess_images(self,all_img_names):
        self.cnn_model.preprocess_images(all_img_names)
    
    #This method is unused now
    def split_train_test(self,train_size=0.8):
        self.img_name_train, self.img_name_test, self.cap_train, self.cap_test = train_test_split(self.all_img_names,
                                                                    self.all_captions,
                                                                    train_size=train_size,
                                                                    random_state=42)
        #self.img_name_test, self.img_name_val, self.cap_test, self.cap_val = train_test_split(self.img_name_rest,
        #                                                            self.cap_rest,
        #                                                            test_size=0.5,
        #                                                            random_state=42)
        print("Total img_name_train = {}".format(len(self.img_name_train)))
        print("Total cap_train = {}".format(len(self.cap_train))) 
        #print("Total img_name_val = {}".format(len(self.img_name_val))) 
        #print("Total cap_val = {}".format(len(self.cap_val))) 
        print("Total img_name_test = {}".format(len(self.img_name_test))) 
        print("Total cap_test = {}".format(len(self.cap_test)))  

    def create_dataset_internal(self,img_names,captions):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = 1000
        #num_steps = len(img_names) // BATCH_SIZE

        def map_func(img_name,cap,img_name_copy):
            img_tensor = np.load(img_name.decode('utf-8')+'.npy')
            return img_tensor,cap,img_name_copy
       
        dataset = tf.data.Dataset.from_tensor_slices((img_names, captions,img_names))
       
        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1,item2,item3: tf.numpy_function(
                map_func, [item1, item2,item3], [tf.float32, tf.int32,tf.string]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    def create_dataset(self):
        #self.split_train_test()
        self.dataset_train=self.create_dataset_internal(self.all_img_names_train,self.all_captions_train)
        print("Total dataset_train size = {}".format(len(self.dataset_train)))
        self.dataset_val=self.create_dataset_internal(self.all_img_names_val,self.all_captions_val)
        print("Total dataset_val size = {}".format(len(self.dataset_val)))
        self.dataset_test=self.create_dataset_internal(self.all_img_names_test,self.all_captions_test)
        print("Total dataset_test size = {}".format(len(self.dataset_test)))



    def load(self):
        #Load the dataframe with filename ,index,captions
        self.load_dataframe()
        #Sanitize the captions
       

        #Tokenize the texts
        #This was reason behind stupid overfitting issue.
        #self.tokenize_captions()
        #Preproces the images
        self.preprocess_images()

    def load_create_dataset(self):
        self.load()
        self.show_stats()
        self.create_dataset()    