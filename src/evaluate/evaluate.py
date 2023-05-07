from time import time

###Make it false for your local m/c debugging
COLLAB=False
if COLLAB == True:
    import wandb

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from inference import Inference
from metric import bleu_score, meteor_score_value, rogue_score_value


class Evaluate:
    def __init__(self,transformer_model,model_training_params,imageCaptionDataset, **kwargs):
        super().__init__(**kwargs)
        self.transformer_model=transformer_model
        self.model_training_params=model_training_params
        self.imageCaptionDataset=imageCaptionDataset
        self.inference= Inference(self.transformer_model,self.imageCaptionDataset.cnn_model,self.imageCaptionDataset.captionProcessor)


    
    def dump_image_caption(self):
        image_caption_path=self.model_training_params.test_image_caption_path
        image_caption_csv_path=os.path.join(image_caption_path,'captions_test.csv')
        dataset_test=self.imageCaptionDataset.dataset_test
        print("Dumping image captions to ",image_caption_csv_path)
        df = pd.DataFrame(columns=['index','filename','caption']) 
        df = df.reindex(columns =['index','filename','caption'])
        start_time_accuracy=time()
        bleu_scores=0.0
        index=0
        processed_images=[]
        for step,(img_tensor,captions,image_names) in enumerate(dataset_test):
            for img_tensor_val,image_name in zip(img_tensor,image_names):
                
                image_name_f=tf.get_static_value(image_name).decode('utf8').split('/')[-1]
                if image_name_f in processed_images:
                    continue
                processed_images.append(image_name_f)
                print("Extracting caption for ",image_name_f)
                start_time=time()
                hypothesis,result,attention_weights=self.inference.extract_caption_from_image_tensor(img_tensor_val)
                
                print("Time  taken for  "+image_name_f+": %.2fs" % (time() - start_time))
                df.loc[len(df.index)] = [index, image_name_f, hypothesis] 
                index=index+1
                print("Extraction count ",index)
        #train_accuracy_score= bleu_scores/count
        df.to_csv(image_caption_csv_path)
        print("Time  taken for  dump_image_caption: %.2fs" % (time() - start_time_accuracy))
        #return train_accuracy_score


    def load_img_caption(self):
        image_caption_path=self.model_training_params.test_image_caption_path
        image_caption_csv_path=os.path.join(image_caption_path,'captions_test.csv')
        image_caption_df=pd.read_csv(image_caption_csv_path)
        #print(image_caption_df.head(5))
        return image_caption_df
    
    def calculate_metrics(self):
        df=self.load_img_caption()
        dataset_test=self.imageCaptionDataset.dataset_test
        index=0
        bleu_scores_total=0
        img_count=0
        image_caption_path=self.model_training_params.test_image_caption_path
        image_caption_csv_metrics_path=os.path.join(image_caption_path,'captions_test_metrics.csv')
        
        print("Dumping image metrics to ",image_caption_csv_metrics_path)
        df = pd.DataFrame(columns=['index','filename','caption','references','BLEU','ROUGE1','ROUGEL','METEOR']) 
        df = df.reindex(columns =['index','filename','caption','references','BLEU','ROUGE1','ROUGEL','METEOR'])

      
        image_caption_csv_path=os.path.join(image_caption_path,'captions_test.csv')
        df_caption=pd.read_csv(image_caption_csv_path)
        for step,(img_tensor,captions,image_names) in enumerate(dataset_test):
            for img_tensor_val,image_name in zip(img_tensor,image_names):
                
                image_name_f=tf.get_static_value(image_name).decode('utf8').split('/')[-1]
                print("Extracting caption for with  ",image_name_f)
                print("Extracting caption for with  index ",index)
                start_time=time()
               
                time_taken=time() - start_time
                df_img=df_caption[df_caption['filename'] == image_name_f]
                if df_img.shape[0] == 0:
                    hypothesis=''
                else:
                    hypothesis=df_img.iloc[0].caption    
                if str(hypothesis) ==  'nan':
                    hypothesis=''
                print("caption is ",hypothesis)
                references = self.imageCaptionDataset.load_img_captions_test(image_name_f)
                bleu_refs = [  x[0] for x in references]
                bleu_scores =bleu_score(bleu_refs,hypothesis)
                print("BLEU Score is....",bleu_scores)
                bleu_scores_total+=bleu_scores
                rogue_refs = [  x[0] for x in references]
                rogue_score_value1=rogue_score_value(rogue_refs,hypothesis)
                rouge1=rogue_score_value1['rouge1']
                rougeL=rogue_score_value1['rougeL']
                print("ROUGE Score is....",rogue_score_value1)
                meteor_refs = [  x[0].split() for x in references]
                meteor_hypo=hypothesis.split()
                meteor_score_value1=meteor_score_value(meteor_refs,meteor_hypo)
                print("METEOR Score is....",meteor_score_value1)
                print("Time  taken for caption generation  "+image_name_f+": %.2fs" % (time_taken))
                df.loc[len(df.index)] = [index, image_name_f,hypothesis,references, bleu_scores,rouge1,rougeL,meteor_score_value1] 
                
                #wer,cider , rouge
                img_count+=1    
                index=index+1
        avg_bleu_score=bleu_scores_total/img_count        
        print("Avearage BLEU-4 Score is :::",avg_bleu_score)
        
        df.to_csv(image_caption_csv_metrics_path)
        
                  