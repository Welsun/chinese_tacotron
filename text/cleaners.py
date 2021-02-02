""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from pywubi import wubi

from text_cleaner import remove,keep
from text_cleaner.processor.common import ASCII, ALPHA, DIGIT, GENERAL_PUNCTUATION, SYMBOLS_AND_PUNCTUATION
from text_cleaner.processor.chinese import CHINESE, CHINESE_SYMBOLS_AND_PUNCTUATION
import unicodedata

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def convert_to_wubi(text):
  return ' '.join(wubi(text))

def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text

def chinese_cleaners(text):
  print ("before chinese_cleaners= ", text)
  text = text.encode('UTF-8').decode('UTF-8')
  text = keep (text, [ASCII, GENERAL_PUNCTUATION, SYMBOLS_AND_PUNCTUATION, CHINESE, CHINESE_SYMBOLS_AND_PUNCTUATION])
  text = unicodedata.normalize('NFKC', text)
  text = remove (text, [CHINESE_SYMBOLS_AND_PUNCTUATION])
  text = collapse_whitespace(text)
  print ("after chinese_cleaners= ", text)
  return text

def chinese_wubi_cleaners(text):
  print ("before chinese_cleaners= ", text)
  text = text.encode('UTF-8').decode('UTF-8')
  text = keep (text, [ASCII, GENERAL_PUNCTUATION, SYMBOLS_AND_PUNCTUATION, CHINESE, CHINESE_SYMBOLS_AND_PUNCTUATION])
  text = unicodedata.normalize('NFKC', text)
  text = remove (text, [CHINESE_SYMBOLS_AND_PUNCTUATION])
  text = convert_to_wubi(text)
  text = collapse_whitespace(text)
  print ("after chinese_cleaners= ", text)
  return text
