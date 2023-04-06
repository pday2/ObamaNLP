import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import unicodedata
import re
import stanza
from collections import defaultdict
from textblob import TextBlob
from graphviz import Source
import nltk
from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
word_token = TreebankWordTokenizer()

nlp = stanza.Pipeline(lang="en") # Initialize the default English pipeline

sent4 = 'For everywhere in this country, there are first steps to be taken, there’s new ground to cover, there are more bridges to be crossed.'
sent4 = 'For everywhere in this country, there are first steps to be taken, there are more bridges to be crossed.'
sent4 = 'For everywhere in this country, there are first steps to be taken.'
sent4 = 'For everywhere in this country, there are more bridges to be crossed.'

parser = CoreNLPParser()
sent, = parser.parse_text(sent4)
sent.pretty_print()