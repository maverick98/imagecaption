import pathlib,sys,os
src_path = pathlib.Path(__file__).parents[1]
print(src_path)
sys.path.append(os.path.dirname(src_path))
from tensorflow import train
import tensorflow as tf

#import time
#import gensim

from src.train.train import Train



from src.model.data.image_caption_dataset import ImageCaptionDataset
from src.model.transformer.transformer import Transformer


from src.model.params.model_params_factory import ModelParamsFactory
from src.model.params.model_training_params_factory import ModelTrainingParamsFactory
# src.model.evaluate
from src.evaluate.evaluate import Evaluate



image_path = '../../dataset/images/Flicker8k_Dataset/'
token_file='../../dataset/texts/Flickr8k.token.txt'
data_limit=40455  # We know this from EDA
num_words = 8357  # We know this from EDA 

data_limit=40455
num_words = 100 


caption_max_len=50 #Need to revisit this.


modelParamsFactory = ModelParamsFactory()
model_params=modelParamsFactory.create()

modelTrainingParamsFactory = ModelTrainingParamsFactory()
model_training_params=modelTrainingParamsFactory.create()

model_training_params.epochs=1
model_training_params.test_image_caption_path='../../'


transformer_model=Transformer(model_params)

imageCaptionDataset=ImageCaptionDataset(model_training_params)
imageCaptionDataset.load_create_dataset()
#imageCaptionDataset.clean_vocabulary_size

#train = Train(transformer_model,model_training_params,imageCaptionDataset)
#train.train(imageCaptionDataset.dataset_train,imageCaptionDataset.dataset_test)


#model =transformer_model


MODEL_OUTPUT='../../saved_model'
#tf.saved_model.save( transformer_model, MODEL_OUTPUT, signatures=None, options=None)
loaded_model=tf.saved_model.load( MODEL_OUTPUT, tags=None, options=None)

model =loaded_model


evaluate = Evaluate(model,model_training_params,imageCaptionDataset)
evaluate.dump_image_caption()
#evaluate.load_img_caption()
evaluate.calculate_metrics()