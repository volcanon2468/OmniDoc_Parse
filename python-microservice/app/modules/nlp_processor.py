import spacy
from typing import Dict, List, Optional
import re
from collections import Counter
import warnings

class NLPProcessor:

    def __init__(self, config: Dict):
        self.config = config
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*lemmatizer did not find POS annotation.*')
            self.nlp = spacy.load('en_core_web_sm')
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer', first=True)
        self.patterns = {'email': re.compile('\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b'), 'phone': re.compile('\\b(\\+\\d{1,2}\\s?)?\\(?\\d{3}\\)?[\\s.-]?\\d{3}[\\s.-]?\\d{4}\\b'), 'url': re.compile('https?://(?:[-\\w.]|(?:%[\\da-fA-F]{2}))+[^\\s]*'), 'date': re.compile('\\b\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4}\\b|\\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \\d{1,2},? \\d{4}\\b')}
        self.doc_keywords = self.config.get('doc_classes', {})

    def _extract_entities(self, doc) -> Dict[str, List[Dict]]:
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append({'text': ent.text, 'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_})
        return entities

    def _extract_patterns(self, text: str) -> Dict[str, List[str]]:
        matches = {}
        for pattern_name, pattern in self.patterns.items():
            matches[pattern_name] = pattern.findall(text)
        return matches

    def _classify_document(self, text: str) -> Dict[str, float]:
        text = text.lower()
        scores = {}
        for doc_type, keywords in self.doc_keywords.items():
            count = sum((1 for keyword in keywords if keyword in text))
            scores[doc_type] = count / len(keywords)
        if not scores:
            return {'type': 'unknown', 'confidence': 0.0, 'scores': {}}
        max_score = max(scores.values())
        if max_score > 0:
            doc_type = max(scores.items(), key=lambda x: x[1])[0]
        else:
            doc_type = 'unknown'
        return {'type': doc_type, 'confidence': max_score, 'scores': scores}

    def _extract_keywords(self, doc, top_n: int=10) -> List[Dict]:
        words = [token.text.lower() for token in doc if not token.is_stop and (not token.is_punct) and (len(token.text) > 2)]
        word_freq = Counter(words)
        total_words = len(words)
        keywords = []
        if total_words == 0:
            return keywords
        for word, count in word_freq.most_common(top_n):
            keywords.append({'word': word, 'count': count, 'score': count / total_words})
        return keywords

    def _calculate_statistics(self, doc) -> Dict:
        try:
            sentences = list(doc.sents)
        except ValueError:
            sentences = [doc]
        words = [token for token in doc if not token.is_punct]
        avg_word_length = sum((len(word.text) for word in words)) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        return {'sentence_count': len(sentences), 'word_count': len(words), 'unique_words': len(set((token.text.lower() for token in words))), 'avg_word_length': round(avg_word_length, 2), 'avg_sentence_length': round(avg_sentence_length, 2), 'char_count': sum((len(token.text) + 1 for token in doc)) - 1}

    def process(self, text: str, extract_entities: bool=True, extract_keywords: bool=True, classify_document: bool=True) -> Dict:
        try:
            if not text or not text.strip():
                raise ValueError('Empty or whitespace-only text provided')
            doc = self.nlp(text)
            entities = self._extract_entities(doc) if extract_entities else {}
            patterns = self._extract_patterns(text)
            classification = self._classify_document(text) if classify_document else {}
            keywords = self._extract_keywords(doc) if extract_keywords else []
            statistics = self._calculate_statistics(doc)
            return {'entities': entities, 'patterns': patterns, 'classification': classification, 'keywords': keywords, 'statistics': statistics}
        except Exception as e:
            raise Exception(f'Error in NLP processing: {str(e)}')