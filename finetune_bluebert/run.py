import torch.nn as nn

import bluebert.model
import bluebert.tokenizer


def run():
    
    model:nn.Module = bluebert.model.getModel()
    tokenizer:nn.Module = bluebert.tokenizer.getTokenizer()
    
    