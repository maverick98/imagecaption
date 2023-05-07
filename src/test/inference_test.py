import pathlib,sys,os
src_path = pathlib.Path(__file__).parents[1]
print(src_path)
sys.path.append(os.path.dirname(src_path))
import tensorflow as tf


import numpy as np
#import time
import os
#import gensim




from src.model.data.image_caption_dataset import ImageCaptionDataset
from src.model.transformer.transformer import Transformer
from src.evaluate.inference import Inference
from src.train.train import Train

from src.model.params.model_params_factory import ModelParamsFactory
from src.model.params.model_training_params_factory import ModelTrainingParamsFactory


caption_max_len=33 #Need to revisit this.
image_path = '../../dataset/images/Flicker8k_Dataset/'

token_file='../../dataset/texts/Flickr8k.token.txt'
token_file_train='../../dataset/texts/Flickr_8k.trainImages.txt'
token_file_val='../../dataset/texts/Flickr_8k.devImages.txt'
token_file_test='../../dataset/texts/Flickr_8k.testImages.txt'
data_limit_train=40455  # We know this from EDA
num_words = 8357  # We know this from EDA 

data_limit_train=10
data_limit_val=5
data_limit_test=5
#num_words = 100 
modelParamsFactory = ModelParamsFactory()
model_params=modelParamsFactory.create()

modelTrainingParamsFactory = ModelTrainingParamsFactory()
model_training_params=modelTrainingParamsFactory.create()
model_training_params.caption_max_len=caption_max_len
model_training_params.image_path=image_path
model_training_params.token_file=token_file
model_training_params.token_file_train=token_file_train
model_training_params.token_file_val=token_file_val
model_training_params.token_file_test=token_file_test
model_training_params.data_limit_train=data_limit_train
model_training_params.data_limit_val=data_limit_val
model_training_params.data_limit_test=data_limit_test
model_training_params.num_words=num_words
model_training_params.epochs=2

transformer_model=Transformer(model_params)

model_training_params
imageCaptionDataset=ImageCaptionDataset(model_training_params,model_params)
imageCaptionDataset.load_create_dataset()



train_module = Train(transformer_model,model_training_params,imageCaptionDataset)
train_module.train(imageCaptionDataset.dataset_train,imageCaptionDataset.dataset_val)


model =transformer_model


MODEL_OUTPUT='../../saved_model'
tf.saved_model.save( transformer_model, MODEL_OUTPUT, signatures=None, options=None)
loaded_model=tf.saved_model.load( MODEL_OUTPUT, tags=None, options=None)

#inference= Inference(model,imageCaptionDataset.cnn_model,imageCaptionDataset.captionProcessor)
#rnd_image_path=os.path.join(image_path,'10815824_2997e03d76.jpg')
#caption,result,attention_weights=inference.extract_caption(rnd_image_path,False)
#print(caption)


model =loaded_model
inference= Inference(model,imageCaptionDataset.cnn_model,imageCaptionDataset.captionProcessor)
rnd_image_path=os.path.join(image_path,'10815824_2997e03d76.jpg')
caption,result,attention_weights=inference.extract_caption(rnd_image_path,False)
print(caption)