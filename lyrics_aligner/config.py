"""
Phonetic dictionaries, stop lists, and acoustic configuration constants.
"""

FUNCTION_WORDS = {
    'a', 'an', 'the', 'to', 'in', 'on', 'at', 'by', 'for', 'of', 'with',
    'and', 'or', 'but', 'if', 'as', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'it', 'its', "it's", 'i', "i'm", "i've", "i'll",
    'you', "you're", "you've", "you'll", 'he', "he's", 'she', "she's",
    'they', "they're", 'we', "we're", 'my', 'your', 'his', 'her', 'our',
    'their', 'me', 'us', 'them', 'that', 'this', 'no', 'so', 'do', "don't",
    'did', "didn't", 'not', 'can', "can't", 'won', "won't", 'from', 'up',
    'out', 'off', 'then', 'than', 'into', 'just', 'like', 'got', 'too',
    'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'no', 'na', 'se', 'me', 'te',
}

DIPHTHONGS = {
    'ai', 'ay', 'ea', 'ee', 'ei', 'ey', 'ie', 'oa', 'oe',
    'oi', 'oy', 'oo', 'ou', 'ow', 'au', 'aw', 'igh', 'eigh'
}

PLOSIVES = {'p', 'b', 't', 'd', 'k', 'g'}
FRICATIVES = {'s', 'z', 'f', 'v', 'th', 'sh', 'ch', 'j', 'h'}
SONORANTS = {'m', 'n', 'l', 'r', 'w', 'y'}
OPEN_VOWELS = {'a', 'o', 'aw', 'ah', 'oh'}

DEFAULT_SR = 16000
DEFAULT_FRAME_MS = 20
DEFAULT_HOP_MS = 10
