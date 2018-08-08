import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import dsdag
from dsdag import core
