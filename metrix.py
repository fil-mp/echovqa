import math
from utils import *
from glossary import *


def bleu(candidate, references, n, weights):
    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is bigger than n)"
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is smaller than n)"
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return str(bleu_result)


def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


def calculate_exactmatch(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return 0
    else:
        return count / total


def similarity_candidate_prediction(candidate_answer, prediction):
    candidate_answer = split_sentence(candidate_answer, 1)

    count = 0
    total = 0
    for word in prediction:
        if word in candidate_answer:
            count += 1
    total = len(candidate_answer)

    if total == 0:
        return 0.0
    else:
        return count / total


def argmax(lst):
    return lst.index(max(lst))


def calculate_appearance_with_normalization(prediction, reference, candidate_set):
    prediction = normalize_word(prediction)
    reference = normalize_word(reference)
    prediction_words = split_sentence(prediction, 1)
    reference_words = split_sentence(reference, 1)

    candidate_set = candidate_set['0']

    similarity_list = []
    candidate_answer_normalized_list = []
    for candidate_answer in candidate_set:
        if isinstance(candidate_answer, int):
            candidate_answer = str(candidate_answer)
        candidate_answer = normalize_word(candidate_answer)
        candidate_answer_normalized_list.append(candidate_answer)
        similarity_list.append(similarity_candidate_prediction(candidate_answer, prediction_words))

    final_prediction = candidate_answer_normalized_list[argmax(similarity_list)]

    if final_prediction == reference:
        return 1.0
    else:
        return 0.0


def calculate_f1score(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0, 0, 0
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall


import re

_period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
_comma_strip = re.compile(r"(\d)(,)(\d)")
_punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
_articles = {"a", "an", "the"}
_manual_map = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_contractions = dict(contractions)


def processPunctuation(inText):
    outText = inText
    for p in _punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(_comma_strip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = _period_strip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = _manual_map.setdefault(word, word)
        if word not in _articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in _contractions:
            outText[wordId] = _contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer_pefomed(answer: str) -> str:
    answer = str(answer)
    answer = processDigitArticle(processPunctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray').replace('\n', ' ').replace('\t', ' ')
    return answer


def calculate_exactmatch_pefomed(pred: str, gt: str) -> float:
    pred_norm = preprocess_answer_pefomed(pred)
    gt_norm = preprocess_answer_pefomed(gt)
    return 1.0 if pred_norm == gt_norm else 0.0
