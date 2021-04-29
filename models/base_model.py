# encoding = utf8
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

class BaseModel(object):
    def __init__(self, config):
        self.config = config