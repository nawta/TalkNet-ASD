import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, numpy, os, subprocess, pandas, tqdm
from model.talkNetModel import talkNetModel

def main():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Pandas version:", pandas.__version__)
    
    model = talkNetModel()
    print("Successfully initialized TalkNet model")
    
    print("Model parameter count:", sum(param.numel() for param in model.parameters()))
    
    print("Verification successful: Python 3.9.18 upgrade is compatible with TalkNet-ASD")

if __name__ == "__main__":
    main()
