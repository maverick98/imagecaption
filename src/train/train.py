from tensorflow import train,  GradientTape, function
from time import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow import reduce_sum,  equal, argmax
from tensorflow import   argmax
from tensorflow import  cast, float32,int32
import math
from src.model.transformer.learning_rate_schedule import LRScheduler
from src.model.metric.metric import  wer_value_tensor
from tensorflow.keras.losses import SparseCategoricalCrossentropy
###Make it false for your local m/c debugging
COLLAB=True
if COLLAB == True:
    import wandb
from src.model.metric.metric import bleu_score_tensor
from src.evaluate.inference import Inference
import tensorflow as tf
import pandas as pd
import os
from collections import OrderedDict

class Train:
    def __init__(self,transformer_model,model_training_params,imageCaptionDataset, **kwargs):
        super().__init__(**kwargs)
        self.transformer_model=transformer_model

        ########################Model Training params###################
        self.model_training_params=model_training_params
        self.imageCaptionDataset=imageCaptionDataset
        self.inference= Inference(self.transformer_model,self.imageCaptionDataset.cnn_model,self.imageCaptionDataset.captionProcessor)
        
        beta_1=self.model_training_params.beta_1
        beta_2=self.model_training_params.beta_2
        epsilon=self.model_training_params.epsilon
        d_model=self.transformer_model.model_params.d_model
        self.optimizer=Adam(LRScheduler(d_model),beta_1,beta_2,epsilon)
        self.checkpoints_path=self.model_training_params.checkpoints_path
        self.max_to_keep=self.model_training_params.max_to_keep
        self.train_image_caption_path=self.model_training_params.train_image_caption_path
        self.test_image_caption_path=self.model_training_params.test_image_caption_path
        self.train_loss=Mean(name='train_loss')
        self.train_accuracy=Mean(name='train_accuracy')
        self.train_wer=Mean(name='train_wer')
        self.val_loss=Mean(name='val_loss')
        self.val_accuracy=Mean(name='val_accuracy')
        self.val_wer=Mean(name='val_wer')

        def wer_function(real,pred):
            #print("real type is ",type(real))
            #print("prediction type is ",type(pred))

            #print("real is ",real)
            #print("prediction is ",pred)

            hypothesis=cast(argmax(pred,axis=2),float32)
            return wer_value_tensor(real,hypothesis)
        self.wer_function=wer_function
        def accuracy_function(real,pred):
            #print("real is ",real)
            #print("prediction is ",pred)

            mask = tf.math.logical_not(equal(real,0))
            accuracy=equal(real,cast(argmax(pred,axis=2),int32))
            accuracy=tf.math.logical_and(mask,accuracy)
            mask = cast(mask,float32)
            accuracy = cast(accuracy,float32)
            return reduce_sum(accuracy)/reduce_sum(mask)
        def accuracy_functionOld(real,pred):
            #print("real is ",real.shape)
            #print("prediction is ",pred.shape)
            hypothesis=cast(argmax(pred,axis=2),float32)
            
            accuracy_score=bleu_score_tensor(references_tensor=real,hypothesis_tensor=hypothesis)
            #print("bleu score is ",accuracy_score)
            #bleu_score(references=tf.ev(real),hypothesis=tf.eval(hypothesis))    
            return accuracy_score
        self.accuracy_function=accuracy_function
        #
        # The below function is taken from https://www.tensorflow.org/tutorials/text/image_captioning
        #
        def loss_function(real, pred):  
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(real, pred)

            mask = (real != 0) & (loss < 1e8) 
            mask = tf.cast(mask, loss.dtype)

            loss = loss*mask
            loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
            return loss

        def loss_functionOld(real, pred):
            #print("real is ",real)
            #print("prediction is ",pred)

            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            loss_ = loss_object(real, pred)
            mask = cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return reduce_sum(loss_)/reduce_sum(mask)
        self.loss_function=loss_function
        ckpt=train.Checkpoint(model=self.transformer_model,optimizer=self.optimizer)
        self.ckpt_mgr=train.CheckpointManager(ckpt,self.checkpoints_path,max_to_keep=self.max_to_keep)


        
        ########################Model Training params###################
    def buildConfig(self):
        self.config={}
        model_params=self.transformer_model.model_params
        self.config['target_vocab_size']=model_params.target_vocab_size
        self.config['max_pos_encoding']=model_params.max_pos_encoding
        self.config['h']=model_params.h
        self.config['d_k']=model_params.d_k
        self.config['d_ff']=model_params.d_ff
        self.config['d_model']=model_params.d_model
        self.config['num_layers']=model_params.num_layers
        self.config['dropout_rate']=model_params.dropout_rate
        


        self.config['epochs']=self.model_training_params.epochs
        self.config['beta_1']=self.model_training_params.beta_1
        self.config['beta_2']=self.model_training_params.beta_2
        self.config['epsilon']=self.model_training_params.epsilon
        self.config['caption_max_len']=self.model_training_params.caption_max_len
        self.config['data_limit_train']=self.model_training_params.data_limit_train
        self.config['data_limit_val']=self.model_training_params.data_limit_val
        self.config['data_limit_test']=self.model_training_params.data_limit_test
        self.config['image_path']=self.model_training_params.image_path

    
    #Let's not do Eager execution which is the default of TensorFlow 2.0
    #@function annotation initiates Graph Execution. Thus it speeds up the computations for large dataset like ours
    
    @function
    def train_step(self,img_tensor, tar_inp,tar_real,image_names):
        with GradientTape() as tape:
            predictions, _ = self.transformer_model(img_tensor, tar_inp,True)
            loss = self.loss_function(tar_real, predictions)
            accuracy_score=self.accuracy_function(tar_real,predictions)
            wer=self.wer_function(tar_real,predictions)

        gradients = tape.gradient(loss, self.transformer_model.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer_model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_score)
        self.train_wer(wer)
    def dump_image_caption(self,image_caption_path,dataset_train_test):
        print("Dumping image captions to ",image_caption_path)
        df = pd.DataFrame(columns=['index','filename','caption']) 
        df = df.reindex(columns =['index','filename','caption'])
        start_time_accuracy=time()
        bleu_scores=0.0
        index=0
        for step,(img_tensor,captions,image_names) in enumerate(dataset_train_test):
            for img_tensor_val,image_name in zip(img_tensor,image_names):
                
                image_name_f=tf.get_static_value(image_name).decode('utf8').split('/')[-1]
                print("Extracting caption for ",image_name_f)
                start_time=time()
                hypothesis,result,attention_weights=self.inference.extract_caption_from_image_tensor(img_tensor_val)
                
                print("Time  taken for  "+image_name_f+": %.2fs" % (time() - start_time))
                df.loc[len(df.index)] = [index, image_name_f, hypothesis] 
                index=index+1
        #train_accuracy_score= bleu_scores/count
        df.to_csv(image_caption_path)
        print("Time  taken for  dump_image_caption: %.2fs" % (time() - start_time_accuracy))
        #return train_accuracy_score     

    def dump_image_caption_epoch(self,epoch,caption_path,dataset_train_tet):
        image_caption_path=os.path.join(caption_path, "caption_epoch_"+str(epoch)+".txt")
        self.dump_image_caption(image_caption_path,dataset_train_tet)

    def dump_image_caption_epoch_train(self,epoch,dataset_train):
        self.dump_image_caption_epoch(epoch,self.train_image_caption_path,dataset_train)  

    def dump_image_caption_epoch_val(self,epoch,dataset_val):
        self.dump_image_caption_epoch(epoch,self.val_image_caption_path,dataset_val)  

        


    def run_train_epoch_step(self,epoch,step,img_tensor,captions,image_names):
        #print(f"Epoch {epoch+1} Step {step} image_names of size {len(image_names)} ")     
        encoder_input=img_tensor
        decoder_input=captions[:,:-1]
        decoder_output=captions[:,1:]
        self.train_step(encoder_input,decoder_input,decoder_output,image_names)

    

    def run_train(self,epoch,train_dataset):
        print("\nTraining on train_dataset of batch size :",len(train_dataset))     
        for step,(img_tensor,captions,image_names) in enumerate(train_dataset):
            self.run_train_epoch_step(epoch,step,img_tensor,captions,image_names)
            if step %100 ==0:
                print(f"Epoch {epoch+1} Step {step} Loss {self.train_loss.result():.4f} WordErrorRate {self.train_wer.result():.4f}  Accuracy {self.train_accuracy.result():.4f}")
                #print(f"Epoch {epoch+1} Step {step} Loss {self.train_loss.result():.4f}   Accuracy {self.train_accuracy.result():.4f}")

    def run_val_step(self,step,img_tensor,captions,image_names):
        encoder_input=img_tensor
        decoder_input=captions[:,:-1]
        decoder_output=captions[:,1:]
        predictions, _ = self.transformer_model(encoder_input, decoder_input,False)
        loss = self.loss_function(decoder_output, predictions)
        accuracy_score=self.accuracy_function(decoder_output,predictions)
        wer=self.wer_function(decoder_output,predictions)
        self.val_loss(loss)
        self.val_accuracy(accuracy_score)
        self.val_wer(wer)

    def run_val_epoch_step(self,epoch,step,img_tensor,captions,image_names):
        #print(f"Epoch {epoch+1} Step {step} image_names of size {len(image_names)} ")       
        self.run_val_step(step,img_tensor,captions,image_names)
        if step %100 ==0:
            #print(f"Epoch {epoch+1} Step {step} Loss {self.val_loss.result():.4f}  ") 
            print(f"Epoch {epoch+1} Step {step} Loss {self.val_loss.result():.4f} WordErrorRate {self.val_wer.result():.4f} Accuracy {self.val_accuracy.result():.4f} ")    
            #print(f"Epoch {epoch+1} Step {step} Loss {self.val_loss.result():.4f}  Accuracy {self.val_accuracy.result():.4f} ")    
    def run_val(self,epoch,val_dataset):    
        print("\nValidating on val_dataset of batch size :",len(val_dataset))     
        for step,(img_tensor,captions,image_names) in enumerate(val_dataset):
            self.run_val_epoch_step(epoch,step,img_tensor,captions,image_names)
    
    def calculate_train_test_accuracy(self,train_dataset,test_dataset):
        print("\nCalculating train accuracy:")     
        train_accuracy_score=self.calculate_accuracy(train_dataset)
        print("\nCalculating test accuracy:")     
        test_accuracy_score=self.calculate_accuracy(test_dataset)
        return train_accuracy_score,test_accuracy_score
    def print_dump_epoch_loss_metrics(self,epoch):
        #Print epoch number and loss value at the end of every epoch
        print(f"Epoch {epoch+1}: Training Loss {self.train_loss.result():.4f}, "
                f"Training WordErrorRate {self.train_wer.result():.4f},"
                f"Training Accuracy {self.train_accuracy.result():.4f},"
                + f"Validation Loss {self.val_loss.result():.4f},"
                + f"Validation WordErrorRate {self.val_wer.result():.4f},"
                + f"Validation Accuracy {self.val_accuracy.result():.4f},"
                )
        #print(f"Epoch {epoch+1}: Training Loss {self.train_loss.result():.4f}, "
        #      
        #        f"Training Accuracy {self.train_accuracy.result():.4f},"
        #        + f"Validation Loss {self.val_loss.result():.4f},"
        #       
        #        + f"Validation Accuracy {self.val_accuracy.result():.4f},"
        #        )
        if COLLAB == True:
            wandb.log({"Epoch" : epoch+1, "Train Loss" : self.train_loss.result(),"Train WordErrorRate" : self.train_wer.result(), "Train Accuracy" : self.train_accuracy.result(), "Validation Loss" : self.val_loss.result(),"Validation WordErrorRate" : self.val_wer.result(),"Validation Accuracy" : self.val_accuracy.result()})          
            #wandb.log({"Epoch" : epoch+1, "Train Loss" : self.train_loss.result(), "Train Accuracy" : self.train_accuracy.result(), "Validation Loss" : self.val_loss.result(),"Validation Accuracy" : self.val_accuracy.result()})          
    def print_dump_accuracy_metrics(self,train_accuracy_score,test_accuracy_score):
        print(  f"Training Accuracy {train_accuracy_score:.4f},"
                + f"Test Accuracy {test_accuracy_score:.4f}"                
            )
        if COLLAB == True:
            wandb.log({"Training Accuracy" : train_accuracy_score, "Test Accuracy" : test_accuracy_score})     
        

    def run_epoch(self,epoch,train_dataset,val_dataset):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.train_wer.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        self.val_wer.reset_states()
        print("\nStart of epoch %d " %(epoch+1))
        print("\n\nTraining:")  
        self.run_train(epoch,train_dataset)
        print("\nValidating:")  
        self.run_val(epoch,val_dataset)
        #self.dump_image_caption_epoch_train(epoch+1,train_dataset)
        #self.dump_image_caption_epoch_val(epoch+1,val_dataset)
        self.print_dump_epoch_loss_metrics(epoch)
        
        
        
        ## Save a checkpoint after every five epochs
        #if (epoch + 1) % 5 == 0:
        #    save_path = self.ckpt_mgr.save()
        #    print(f"Saved checkpoint at epoch {epoch+1}")
        print("\nEnd of epoch %d " %(epoch+1))
    def check_monotonocity_by_delta(self,nums,min_delta=0.001):
        delta_count=0
        for i in range(len(nums)-1):
            delta=abs(nums[i]-nums[i+1])
            
            if delta >= min_delta or math.isclose(delta,min_delta):
                delta_count=delta_count+1

        if delta_count ==  len(nums) -1:
            return True
        return False  
    def run_epochs(self,train_dataset,val_dataset):
        min_epoch_val_loss=1000.0 # Take a big number
        max_epoch_val_accuracy=0.0 # Take a small number
        monitor=self.model_training_params.monitor
        min_delta=self.model_training_params.min_delta
        patience=self.model_training_params.patience
        model_output=self.model_training_params.model_output
        print("Early Stopping Criteria monitor::",monitor)
        print("Early Stopping Criteria min_delta::",min_delta)
        print("Early Stopping Criteria patience::",patience)
        print("Early Stopping Criteria model_outout::",model_output)
        patience_map = OrderedDict()
        for epoch in range(self.model_training_params.epochs):
            self.run_epoch(epoch,train_dataset,val_dataset)
            cur_epoch_val_loss=self.val_loss.result()
            cur_epoch_val_accuracy=self.val_accuracy.result()
            patience_map[epoch]={'val_loss':cur_epoch_val_loss,'val_accuracy':cur_epoch_val_accuracy}
            
            if monitor == 'val_loss':
                if  cur_epoch_val_loss < min_epoch_val_loss :
                    min_epoch_val_loss=cur_epoch_val_loss
                    if epoch >0:
                        print("\nSaving the model of  epoch %d " %(epoch+1))
                        start_save_time=time()
                        tf.saved_model.save(self.transformer_model, self.model_training_params.model_output, signatures=None, options=None)
                        print("Total time taken to save model: %.2fs" % (time() - start_save_time)) 
            if len(patience_map) > patience:
                nums=[]
                for key_epoch in reversed(patience_map):             
                    if len(nums) < patience :
                        nums.append(patience_map[key_epoch][monitor])

                if self.check_monotonocity_by_delta(nums,min_delta) == False:
                    print("\nFor last  ",(epoch+1)," epoch(s)" ", there were no improvement more than  ",min_delta)
                    print("\nStopping the training...")
    
    
    def train(self,train_dataset,val_dataset):
        self.buildConfig()
        print("Model Params are",self.transformer_model.model_params.toJSON())
        print("Model Training Params are",self.model_training_params.toJSON())
        if COLLAB == True:
            
            wandb.init(project="Group4CDS", entity="group4cds", name="group4cds",config=self.config)
        start_time=time()
        self.run_epochs(train_dataset,val_dataset)
        print("Total time taken: %.2fs" % (time() - start_time)) 
        if COLLAB == True:
            wandb.finish()
