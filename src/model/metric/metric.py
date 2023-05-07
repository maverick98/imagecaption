from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from jiwer import wer

import tensorflow as tf
chencherry = SmoothingFunction()

bleu1_weights=(1.0,0,0,0)
bleu2_weights=(0.5,0.5,0,0)
bleu3_weights=(0.3,0.3,0.3,0)
bleu4_weights=(0.25,0.25,0.25,0.25)


def bleu_score_tensor(references_tensor,hypothesis_tensor,n=4):
    
    if references_tensor is None or hypothesis_tensor is None:
        return 0.0
    if tf.get_static_value(references_tensor) is None or tf.get_static_value(hypothesis_tensor) is None:
        return 0.0
    references=tf.get_static_value(references_tensor)
    hypothesis=tf.get_static_value(hypothesis_tensor)[0]
    return bleu_score(references,hypothesis)


def bleu_score(references,hypothesis,n=4):
  
    if references is None or hypothesis is None:
        return 0.0
    #print('references ',references)
    #print('hypothesis ',hypothesis)
    #print('references type ',type(references))
    #print('hypothesis type ',type(hypothesis))
    #references=references.split()
    #hypothesis=hypothesis.split()
    weights=bleu4_weights
    if n==1:
        weights=bleu1_weights
    elif n==2:
        weights=bleu2_weights
    elif n==3:
        weights=bleu3_weights 
    elif n==4:
        weights=bleu4_weights            
    return sentence_bleu(references, hypothesis,weights=weights,smoothing_function=chencherry.method1)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
def rogue_score_value(references,hypothesis):
    return scorer.score_multi(references,hypothesis)

def meteor_score_value(references,hypothesis):
    score = meteor_score(references,hypothesis)
    return score


def wer_value_tensor(references_tensor,hypothesis_tensor):
    
    if references_tensor is None or hypothesis_tensor is None:
        return 1.0
    if tf.get_static_value(references_tensor) is None or tf.get_static_value(hypothesis_tensor) is None:
        return 1.0
    references=tf.get_static_value(references_tensor)
    hypothesis=tf.get_static_value(hypothesis_tensor)
    references = [ str(x) for x in references ]
    hypothesis = [ str(x) for x in hypothesis ]
    return wer_value(references,hypothesis)

def wer_value(references,hypothesis):
    error = wer(references,hypothesis)
    return error
