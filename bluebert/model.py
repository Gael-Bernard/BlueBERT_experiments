from transformers.models.auto import AutoModelForTokenClassification #type: ignore

import torch
import torch.nn as nn


def getOriginalModel() -> AutoModelForTokenClassification :
    """Gets a version of BlueBERT for another task on HuggingFace

    Returns:
        AutoModelForTokenClassification: BlueBERT
    """
    model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained("bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16")
    with open("cache/model.pth", "wb") as file:
        torch.save(model, file)
    return model



def getModelFromTorch(path:str) -> 'nn.Module|None':
    """Retrieves the version of BlueBERT saved at the given path under Torch format

    Args:
        path (str): Path to the torch-saved model module

    Returns:
        nn.Module | None : the model module, if found at given location
    """
    model: 'nn.Module|None'
    try:
        model = torch.load(path)
    except:
        model = None
    return model


def getModel() -> nn.Module:
    """Tries getting the model from cache and downloads it if it is not cached

    Returns:
        nn.Module: BlueBERT
    """
    
    cachedModule: 'nn.Module|None' = getModelFromTorch("cache/model.pth")
    module: 'nn.Module'
    
    if cachedModule == None:
        module = getOriginalModel() #type: ignore
    else:
        module = cachedModule 
    return module