#This file should not have been created at first place.
#It is too late to extend evaluate.py to add dependency injection
#Watch out of DI fwk in python

import pandas as pd
from time import time
import os
from metric import bleu_score, meteor_score_value, rogue_score_value

class EvaluateSequential:
    def __init__(self,imageCaptionDataset,**kwargs):
        super().__init__(**kwargs)
        self.imageCaptionDataset=imageCaptionDataset
    def dump_image_caption(self):
        image_caption_path=self.imageCaptionDataset.datasetInput.test_image_caption_path
        image_caption_csv_path=os.path.join(image_caption_path,'captions_test_lstm.csv')
        
        print("Dumping image captions to ",image_caption_csv_path)
        df = pd.DataFrame(columns=['index','filename','caption']) 
        df = df.reindex(columns =['index','filename','caption'])
        start_time_accuracy=time()
        bleu_scores=0.0
        index=0
         
        
        encoding_test=self.imageCaptionDataset.load_encoded_test_images()
        index=0
        for key in encoding_test:
            image = encoding_test[key].reshape((1,2048))
            start_time=time()
            hypothesis=self.imageCaptionDataset.greedySearch(image)
            print("Time  taken for  "+key+": %.2fs" % (time() - start_time))
            df.loc[len(df.index)] = [index, key, hypothesis] 
            index=index+1
            print("Extraction count ",index)
        df.to_csv(image_caption_csv_path)
        print("Time  taken for  dump_image_caption: %.2fs" % (time() - start_time_accuracy))  

    def load_img_caption(self):
        image_caption_path=self.imageCaptionDataset.datasetInput.test_image_caption_path
        image_caption_csv_path=os.path.join(image_caption_path,'captions_test_lstm.csv')

        image_caption_df=pd.read_csv(image_caption_csv_path)
        #print(image_caption_df.head(5))
        return image_caption_df
    def calculate_metrics(self):
        df=self.load_img_caption()
       
        index=0
       
        bleu_scores_total=0
        img_count=0
        image_caption_path=self.imageCaptionDataset.datasetInput.test_image_caption_path
        image_caption_csv_path=os.path.join(image_caption_path,'captions_test_lstm.csv')
     
        image_caption_csv_metrics_path=os.path.join(image_caption_path,'captions_test_metrics_lstm.csv')
        
        print("Dumping image metrics to ",image_caption_csv_metrics_path)
        df = pd.DataFrame(columns=['index','filename','caption','references','BLEU','ROUGE1','ROUGEL','METEOR']) 
        df = df.reindex(columns =['index','filename','caption','references','BLEU','ROUGE1','ROUGEL','METEOR'])

      
      
        df_caption=pd.read_csv(image_caption_csv_path)
        for key, desc_list in self.imageCaptionDataset.test_descriptions.items():
            image_name_f=key+'.jpg'
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
            references = self.imageCaptionDataset.test_descriptions_raw[key]
            print("references are ",references)
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
        print("Avearage BLEU-4 Score using CNN-LSTM is :::",avg_bleu_score)
        
        df.to_csv(image_caption_csv_metrics_path)
        
                   
          
    
    

    