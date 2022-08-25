
import nltk
def get_bleu_score(reference, hypothesis):
    '''
    Args:
    - reference: target sentences (batch_size x seq_len)
    - hypothesis: predicted sentences (batch_size x seq_len)
    Return: 
    - bleu_score: corpos bleu score between reference and hypothesis 
    '''

    # remove special tokens 
    new_reference = []
    for sentence in reference:
        new_sentence=[]
        for word in sentence:
            if word == 'EOS':
                break
            if word != 'BOS' and word != 'EOS' and word != 'PAD':
                new_sentence.append(word)
        new_reference.append([new_sentence])

    new_hypothesis = []
    for sentence in hypothesis:
        new_sentence = []
        for word in sentence:
            if word == 'EOS':
                break
            if word != 'BOS' and word != 'EOS' and word != 'PAD':
                new_sentence.append(word)
        new_hypothesis.append(new_sentence)
    
    # calculate BLEU score 
    bleu1 = nltk.translate.bleu_score.corpus_bleu(new_reference, new_hypothesis,weights=([1.0]))
    bleu2 = nltk.translate.bleu_score.corpus_bleu(new_reference, new_hypothesis,weights=(0.5,0.5))
    bleu3 = nltk.translate.bleu_score.corpus_bleu(new_reference, new_hypothesis,weights=(0.33,0.33,0.33))
    bleu4 = nltk.translate.bleu_score.corpus_bleu(new_reference, new_hypothesis,weights=(0.25,0.25,0.25,0.25))
    smoothingfunction = nltk.translate.bleu_score.SmoothingFunction().method1
    bleu_smooth = nltk.translate.bleu_score.corpus_bleu(new_reference, new_hypothesis, smoothing_function=smoothingfunction)

    return (bleu1, bleu2, bleu3, bleu4, bleu_smooth)

'''
reference = [['BOS','i','want','some','ice','cream','.','EOS','PAD']]
hypothesis = [['BOS','i','want','an','ice','cream','.','EOS','PAD']]
bleu1, bleu2, bleu3, bleu4 = get_bleu_score(reference, hypothesis)
print('BLEU-1: ', bleu1)
print('BLEU-2: ', bleu2)
print('BLEU-3: ', bleu3)
print('BLEU-4: ', bleu4)
'''