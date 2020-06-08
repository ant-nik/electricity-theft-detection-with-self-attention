"""
    Some utils based on source from original repository
    https://github.com/neuralmind-ai/electricity-theft-detection-with-self-attention
"""

import os

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)