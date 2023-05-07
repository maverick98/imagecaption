import sys
import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# print(SCRIPT_DIR)

from src.model.data.cnn_model import CNN_Model
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import PIL
from matplotlib.pylab import plt
from PIL import Image
from src.model.ds.probability_node import ProbabilityNode
from src.model.ds.beam_search import BeamSearch



class Inference:
     def __init__(self,transformer_model,cnn_model,captionProcessor,search_strategy='Greedy',beam_width=2, **kwargs):
        super().__init__(**kwargs)
        self.transformer_model=transformer_model
        self.cnn_model=cnn_model
        self.captionProcessor=captionProcessor
        self.search_strategy=search_strategy
        self.beam_width=beam_width

     def predict(self,img_tensor_val,output,level,curNode):
        #predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = self.transformer_model(img_tensor_val,output,False)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        probability, predicted_ids = tf.nn.top_k(predictions,self.beam_width)
        for predicted_id in predicted_ids:
            probability_node=ProbabilityNode(level,predicted_id)
            curNode.addChild(probability_node)
               
     def perform_beam_search(self,img_tensor_val):
        if len(img_tensor_val.shape) == 2:
            img_tensor_val=tf.expand_dims(img_tensor_val, 0)
        tokenizer=self.captionProcessor.tokenizer
  
        start_token=tokenizer.word_index['<start>']
        end_token=tokenizer.word_index['<end>']
        
        #decoder input is start token.
        decoder_input = [start_token]
        decoder_input = tf.expand_dims(decoder_input, 0) #tokens
        output=tf.keras.preprocessing.sequence.pad_sequences(decoder_input, padding='post', maxlen= self.captionProcessor.caption_max_len)
       
        
        rootNode=ProbabilityNode(0,start_token)
        print('beam width is ',self.beam_width)
        beamSearch=BeamSearch(rootNode,self.beam_width)
        level = 1
        predictions, attention_weights = self.transformer_model(img_tensor_val,output,False)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        probabilities, predicted_ids = tf.nn.top_k(predictions,self.beam_width)
        #for (probability,predicted_id) in zip(probabilities,predicted_ids):
        predicted_ids_value=tf.get_static_value(predicted_ids)
        probabilities_value=tf.get_static_value(probabilities)
        for i in range(self.beam_width):
            predicted_id=predicted_ids_value[0][0][i]
            probability=probabilities_value[0][0][i]
            childNode=ProbabilityNode(level,predicted_id,probability)
            rootNode.addChild(childNode)

        def predictToken(nodeVisited):
            if nodeVisited.isLeafNode() == True:
                if nodeVisited.isStopNode() == True:
                    return

                keysFromRoot=nodeVisited.getKeysFromRoot()
                #print("keysFromRoot is  ",keysFromRoot)
                decoder_input = [start_token]
                decoder_input = tf.expand_dims(decoder_input, 0) #tokens
                output=tf.keras.preprocessing.sequence.pad_sequences(decoder_input, padding='post', maxlen= self.captionProcessor.caption_max_len)
                for i in range(len(keysFromRoot)):
                    if i ==0:
                        continue
                    output[0][i]=keysFromRoot[i]
                #print("Predicting for ",output[0])    
                predictions, attention_weights = self.transformer_model(img_tensor_val,output,False)
                predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
                predicted_ids_value=tf.get_static_value(predicted_ids)
                probabilities_value=tf.get_static_value(probabilities)
                for i in range(self.beam_width):
                    predicted_id=predicted_ids_value[0][0][i]
                    probability=probabilities_value[0][0][i]
                    childNode=ProbabilityNode(nodeVisited.level+1,predicted_id,probability)
                    if predicted_id == end_token:
                        childNode.stopNode=True
                        childNode.output=tf.squeeze(output, axis=0)
                        childNode.attention_weights=attention_weights
                    else:
                        childNode.stopNode=False     
                    
                    nodeVisited.addChild(childNode)
    
        
        for i in range(self.captionProcessor.caption_max_len-1):    
            beamSearch.visit(predictToken)
            shouldStop=beamSearch.prune_nodes()
            if shouldStop == True:
                break

        

        print("Finished Beam search!!!")        

        return beamSearch.calculateHighestProbabilityPath(tokenizer)


        
     def perform_greedy_search(self,img_tensor_val):
        if len(img_tensor_val.shape) == 2:
            img_tensor_val=tf.expand_dims(img_tensor_val, 0)
        tokenizer=self.captionProcessor.tokenizer
  
        start_token=tokenizer.word_index['<start>']
        end_token=tokenizer.word_index['<end>']
        
        #decoder input is start token.
        decoder_input = [start_token]
        decoder_input = tf.expand_dims(decoder_input, 0) #tokens
        output=tf.keras.preprocessing.sequence.pad_sequences(decoder_input, padding='post', maxlen= self.captionProcessor.caption_max_len)
        #output now becomes [[ ]]
        result = [] #word list

        for i in range(self.captionProcessor.caption_max_len-1):
            
        
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer_model(img_tensor_val,output,False)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                return result,tf.squeeze(output, axis=0), attention_weights
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            #print('predicted_id is ',predicted_id)
            #print('int(predicted_id) is ',int(predicted_id))
            if int(predicted_id) < len(tokenizer.index_word):
                #print(tokenizer.index_word[int(predicted_id)])
                result.append(tokenizer.index_word[int(predicted_id)])
            #result.append(prepareDataset.tokenizer.index_word[int(predicted_id)%42])
            #output = tf.concat([output, predicted_id], axis=-1)
            output[0][i+1]=predicted_id

        return result,tf.squeeze(output, axis=0), attention_weights   
     def evaluate(self,img_tensor_val):
        if self.search_strategy == 'Greedy':
            return self.perform_greedy_search(img_tensor_val)
        if self.search_strategy == 'Beam':
            return self.perform_beam_search(img_tensor_val)
        
        return None,None,None
     
     def extract_caption_from_image_tensor(self,img_tensor_val):
        caption_words,result,attention_weights= self.evaluate(img_tensor_val)
        #remove "<unk>" in result
        for i in caption_words:
            if i=="<unk>":
                caption_words.remove(i)
        caption=" ".join(caption_words)        
        return caption,result,attention_weights        
     def extract_caption(self,image_path,plot_image=True):
        #print("image path is ",image_path)
        if plot_image == True:        
            temp_image = np.array(Image.open(image_path))
            plt.imshow(temp_image)
            plt.show() 
        
        temp_input = tf.expand_dims(self.cnn_model.load_image(image_path)[0], 0)
        img_tensor_val = self.cnn_model.extract_feature(temp_input)
        #print("image tensor shape ",img_tensor_val.shape)
        return self.extract_caption_from_image_tensor(img_tensor_val)
        
           

       
      
     