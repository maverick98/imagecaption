import pathlib,sys,os
src_path = pathlib.Path(__file__).parents[1]
print(src_path)
sys.path.append(os.path.dirname(src_path))
from src.model.metric.metric import bleu_score, wer_value,rogue_score_value, meteor_score_value
#from nltk import word_tokenize


references=[[' snow dog in red harness'], [' white dog and brown dog attached by harnesses'], [' white dog with red straps on it is looking back'], [' the dog has red straps in its back'], [' the dogs are joined together with red harness']]
hypothesis="the dog with the green green harness is"

bleu_refs = [  x[0] for x in references]
print(bleu_refs)

print(bleu_score(bleu_refs,hypothesis,1))
print(bleu_score(bleu_refs,hypothesis,2))
print(bleu_score(bleu_refs,hypothesis,3))
print(bleu_score(bleu_refs,hypothesis,4))



rogue_refs = [  x[0] for x in references]
rogue_score_res=rogue_score_value(rogue_refs,hypothesis)
print("ROGUE-1 SCORE ",rogue_score_res['rouge1'])
print("ROGUE-L SCORE ",rogue_score_res['rougeL'])


metero_refs = [  x[0].split() for x in references]

print ("metero_refs is ",metero_refs)
meteor_hypo=hypothesis.split()
print ("meteor_hypo is ",meteor_hypo)
print("METEOR SCORE ",meteor_score_value(metero_refs,meteor_hypo))



#print(meteor_score_value([['this', 'is', 'a', 'cat']], ['non', 'matching', 'hypothesis']))

reference =  ["1","2"]
hypothesis = ["1","2"]

print("wer score:::",wer_value(reference,hypothesis))




