from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import json
import argparse
import string
import re
import tiktoken
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import math
import unicodedata


PATH = {
    'ARC': 'ARC',
    'BIGBench': 'BIGBench',
    'BigCodeBench': 'BigCodeBench',
    'FinQA': 'FinQA',
    'GPQA': 'GPQA',
    'GSM8K': 'GSM8K',
    'HiToM': 'HiToM',
    'IFEval': 'IFEval',
    'LegalBench': 'LegalBench',
    'LogicGame': 'LogicGame',
    'MATH': 'MATH-500',
    'MedQA': 'MedQA',
    'MMLU': 'MMLU',
    'NaturalPlan': 'NaturalPlan',
    'PlanBench': 'PlanBench',
    'PubMedQA': 'PubMedQA',
    'RACE': 'RACE',
    'RAVEN': 'RAVEN',
    'RuleTaker': 'RuleTaker',
    'ScienceQA': 'ScienceQA',
    'SciTLDR': 'SciTLDR',
    'XSum': 'XSum',
    'BigGenBench': 'BigGenBench',
    'LiveBench': 'LiveBench',
    'TableBench': 'TableBench',
    'Flores': 'Flores',
    'MMMLU': 'MMMLU',
    'MMLUPro': 'MMLUPro'
}

MODEL_FROM_API =[
    'xai/grok-2-latest',
]

MODELS_FROM_OPENROUTER = [
    'openai/gpt-4o-2024-11-20'
]

MODELS_WITH_REASONING = [
]
MODEL_PRICE = {
    'openai/gpt-4o-2024-11-20':{
        'input': 2.5,
        'output': 10
    }
}
# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False

def get_token_count(prompt) -> int:
    encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(prompt))


def compute_price(input_prompt, output_token_count, model_id):
    input_token_count = get_token_count(input_prompt)
    try:
        return MODEL_PRICE[model_id]['input'] * input_token_count *1.0/1000000 + MODEL_PRICE[model_id]['output'] * output_token_count *1.0/1000000
    except KeyError:
        raise ValueError(f"Model {model_id} not found in MODEL_PRICE")
    

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            idx = string.rfind('he final answer is:')
            if idx < 0:
                return None
            else:
                retval = string.split("he final answer is:")[-1].strip().replace('$$', '').split("I hope it is correct")[0].split("\n\n")[0].strip()
                if retval.startswith('$'):
                    retval = retval[1:-1]
                return retval, [string.index(retval), string.index(retval) + len(retval)]

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval, [idx, right_brace_idx+1]

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return s

def extract_numbers_from_string(input_string):
    # Use regular expression to find all numbers (including floats and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', input_string)
    # Convert the extracted strings to float or int
    result = [float(num) if '.' in num else int(num) for num in numbers]
    return result

def separate_characters(line):
    return list(line.strip().replace(" ", ""))

def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1] 
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)
    
    return tokenized
    
def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order+1):
            if i+j <= nWords:
                ngram = tuple(wordList[i:i+j])
                counts[j-1][ngram]+=1
   
    return counts

def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)
 
    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])


    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount


def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)
    
    factor = beta**2
    
    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order]/hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order]/reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor*ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1+factor)*ngramPrec[order]*ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16
            
    return ngramF, ngramRec, ngramPrec

def computeChrF(fpRef, fpHyp, nworder, ncorder, beta, sentence_level_scores = None):
    norder = float(nworder + ncorder)

    # initialisation of document level scores
    totalMatchingCount = defaultdict(float)
    totalRefCount = defaultdict(float)
    totalHypCount = defaultdict(float)
    totalChrMatchingCount = defaultdict(float)
    totalChrRefCount = defaultdict(float)
    totalChrHypCount = defaultdict(float)
    averageTotalF = 0.0

    nsent = 0
    for hline, rline in zip(fpHyp, fpRef):
        nsent += 1
        
        # preparation for multiple references
        maxF = 0.0
        bestWordMatchingCount = None
        bestCharMatchingCount = None
        
        hypNgramCounts = ngram_counts(separate_punctuation(hline), nworder)
        hypChrNgramCounts = ngram_counts(separate_characters(hline), ncorder)

        # going through multiple references

        refs = rline.split("*#")

        for ref in refs:
            refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
            refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

            # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
            matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
            matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(refChrNgramCounts, hypChrNgramCounts)
                    
            # n-gram f-scores, recalls and precisions
            ngramF, ngramRec, ngramPrec = ngram_precrecf(matchingNgramCounts, totalRefNgramCount, totalHypNgramCount, beta)
            chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount, beta)

            sentRec  = (sum(chrNgramRec.values())  + sum(ngramRec.values()))  / norder
            sentPrec = (sum(chrNgramPrec.values()) + sum(ngramPrec.values())) / norder
            sentF    = (sum(chrNgramF.values())    + sum(ngramF.values()))    / norder

            if sentF > maxF:
                maxF = sentF
                bestMatchingCount = matchingNgramCounts
                bestRefCount = totalRefNgramCount
                bestHypCount = totalHypNgramCount
                bestChrMatchingCount = matchingChrNgramCounts
                bestChrRefCount = totalChrRefNgramCount
                bestChrHypCount = totalChrHypNgramCount
        # all the references are done


        # write sentence level scores
        if sentence_level_scores:
            sentence_level_scores.write("%i::c%i+w%i-F%i\t%.4f\n"  % (nsent, ncorder, nworder, beta, 100*maxF))


        # collect document level ngram counts
        for order in range(nworder):
            totalMatchingCount[order] += bestMatchingCount[order]
            totalRefCount[order] += bestRefCount[order]
            totalHypCount[order] += bestHypCount[order]
        for order in range(ncorder):
            totalChrMatchingCount[order] += bestChrMatchingCount[order]
            totalChrRefCount[order] += bestChrRefCount[order]
            totalChrHypCount[order] += bestChrHypCount[order]

        averageTotalF += maxF

    # all sentences are done
     
    # total precision, recall and F (aritmetic mean of all ngrams)
    totalNgramF, totalNgramRec, totalNgramPrec = ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount, beta)
    totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = ngram_precrecf(totalChrMatchingCount, totalChrRefCount, totalChrHypCount, beta)

    totalF    = (sum(totalChrNgramF.values())    + sum(totalNgramF.values()))    / norder
    averageTotalF = averageTotalF / nsent
    totalRec  = (sum(totalChrNgramRec.values())  + sum(totalNgramRec.values()))  / norder
    totalPrec = (sum(totalChrNgramPrec.values()) + sum(totalNgramPrec.values())) / norder

    return totalF, averageTotalF, totalPrec, totalRec