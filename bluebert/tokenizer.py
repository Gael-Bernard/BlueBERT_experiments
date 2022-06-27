import os

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer

import torch
import torch.nn as nn


def getOriginalTokenizer() -> PreTrainedTokenizer :
    """Gets a version of the BlueBERT tokenizer for another task on HuggingFace

    Returns:
        AutoModelForTokenClassification: BlueBERT tokenizer
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16") #type: ignore
    os.makedirs("cache", exist_ok=True)
    with open("cache/tokenizer.pth", "wb") as file:
        torch.save(tokenizer, file)
    return tokenizer



def getTokenizerFromTorch(path:str) -> 'nn.Module|None':
    """Retrieves the version of the BlueBERT tokenizer saved at the given path under Torch format

    Args:
        path (str): Path to the torch-saved model tokenizer module

    Returns:
        nn.Module | None : the tokenizer module, if found at given location
    """
    tokenizer: 'nn.Module|None'
    try:
        tokenizer = torch.load(path)
    except:
        tokenizer = None
    return tokenizer


def getTokenizer() -> nn.Module:
    """Tries getting the model tokenizer from cache and downloads it if it is not cached

    Returns:
        nn.Module: BlueBERT tokenizer
    """
    
    cachedTokenizer: 'nn.Module|None' = getTokenizerFromTorch("cache/model.pth")
    tokenizer: 'nn.Module'
    
    if cachedTokenizer == None:
        tokenizer = getOriginalTokenizer() #type: ignore
    else:
        tokenizer = cachedTokenizer 
    return tokenizer