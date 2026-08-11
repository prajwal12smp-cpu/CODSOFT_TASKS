import re
from typing import List

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'could', "couldn't", 'did', "didn't",
    'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
    'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers',
    'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
    "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll",
    "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these',
    'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll",
    "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
    "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',
    "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves'
}


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    return [token for token in cleaned.split() if token]


class PorterStemmer:
    def __init__(self):
        self.vowels = set('aeiou')

    def is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in self.vowels:
            return False
        if ch == 'y':
            if i == 0:
                return True
            return not self.is_consonant(word, i - 1)
        return True

    def measure(self, stem: str) -> int:
        count = 0
        prev_cons = None
        for i in range(len(stem)):
            consonant = self.is_consonant(stem, i)
            if i == 0:
                prev_cons = consonant
            else:
                if prev_cons and not consonant:
                    count += 1
                prev_cons = consonant
        return count

    def contains_vowel(self, word: str) -> bool:
        return any(not self.is_consonant(word, i) for i in range(len(word)))

    def ends_double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and self.is_consonant(word, len(word) - 1)

    def cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        if (not self.is_consonant(word, -1)) or self.is_consonant(word, -2) or (not self.is_consonant(word, -3)):
            return False
        if word[-1] in 'wxy':
            return False
        return True

    def _replace(self, word: str, suffix: str, replacement: str, condition=None):
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if condition is None or condition(base):
                return base + replacement
        return word

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)
        return word

    def _step1a(self, word: str) -> str:
        if word.endswith('sses'):
            return word[:-2]
        if word.endswith('ies'):
            return word[:-2]
        if word.endswith('ss'):
            return word
        if word.endswith('s'):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith('eed'):
            base = word[:-3]
            if self.measure(base) > 0:
                return base + 'ee'
            return word

        for suffix in ('ed', 'ing'):
            if word.endswith(suffix):
                base = word[:-len(suffix)]
                if self.contains_vowel(base):
                    word = base
                    if word.endswith(('at', 'bl', 'iz')):
                        return word + 'e'
                    if self.ends_double_consonant(word) and word[-1] not in 'lsz':
                        return word[:-1]
                    if self.measure(word) == 1 and self.cvc(word):
                        return word + 'e'
                break
        return word

    def _step1c(self, word: str) -> str:
        if word.endswith('y'):
            base = word[:-1]
            if self.contains_vowel(base):
                return base + 'i'
        return word

    def _step2(self, word: str) -> str:
        replacements = {
            'ational': 'ate',
            'tional': 'tion',
            'enci': 'ence',
            'anci': 'ance',
            'izer': 'ize',
            'abli': 'able',
            'alli': 'al',
            'entli': 'ent',
            'eli': 'e',
            'ousli': 'ous',
            'ization': 'ize',
            'ation': 'ate',
            'ator': 'ate',
            'alism': 'al',
            'iveness': 'ive',
            'fulness': 'ful',
            'ousness': 'ous',
            'aliti': 'al',
            'iviti': 'ive',
            'biliti': 'ble',
            'logi': 'log',
        }
        for suffix, replacement in replacements.items():
            if word.endswith(suffix):
                base = word[:-len(suffix)]
                if self.measure(base) > 0:
                    return base + replacement
                return word
        return word

    def _step3(self, word: str) -> str:
        replacements = {
            'icate': 'ic',
            'ative': '',
            'alize': 'al',
            'iciti': 'ic',
            'ical': 'ic',
            'ful': '',
            'ness': '',
        }
        for suffix, replacement in replacements.items():
            if word.endswith(suffix):
                base = word[:-len(suffix)]
                if self.measure(base) > 0:
                    return base + replacement
                return word
        return word

    def _step4(self, word: str) -> str:
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment',
            'ent', 'ion', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize',
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                base = word[:-len(suffix)]
                if self.measure(base) > 1:
                    if suffix == 'ion' and base.endswith(('s', 't')):
                        return base
                    if suffix != 'ion':
                        return base
        return word

    def _step5(self, word: str) -> str:
        if word.endswith('e'):
            base = word[:-1]
            if self.measure(base) > 1 or (self.measure(base) == 1 and not self.cvc(base)):
                word = base
        if self.measure(word) > 1 and self.ends_double_consonant(word) and word.endswith('l'):
            word = word[:-1]
        return word


stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    tokens = tokenize(text)
    filtered = [token for token in tokens if token not in STOP_WORDS]
    return ' '.join(stemmer.stem(token) for token in filtered)


def predict_from_text(message: str, vectorizer, model):
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])
    return model.predict(features)[0]
