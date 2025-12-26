from typing import Set

from graphgen.bases import BaseEvaluator, QAPair
from graphgen.utils import NLTKHelper, detect_main_language


class MTLDEvaluator(BaseEvaluator):
    """
    Metrics for measuring the lexical diversity of text.
    """

    def __init__(self, threshold: float = 0.72):
        self.nltk_helper = NLTKHelper()
        self.stopwords_en: Set[str] = set(self.nltk_helper.get_stopwords("en"))
        self.stopwords_zh: Set[str] = set(self.nltk_helper.get_stopwords("zh"))
        self.threshold = threshold

    def evaluate(self, pair: QAPair) -> float:
        """
        Calculate the MTLD (Mean Token Length Diversity) score for a given text.

        min is 1.0
        higher is better
        """
        text = pair.answer
        if not text or not text.strip():
            return 0.0

        lang = detect_main_language(text)
        tokens = self.nltk_helper.word_tokenize(text, lang)

        stopwords = self.stopwords_zh if lang == "zh" else self.stopwords_en
        filtered_tokens = [word for word in tokens if word not in stopwords]
        filtered_tokens = [word for word in filtered_tokens if word.isalnum()]

        if not filtered_tokens:
            return 0

        # Compute forward factors
        forward_factors = self._compute_factors(filtered_tokens, self.threshold)

        # Compute backward factors
        backward_factors = self._compute_factors(filtered_tokens[::-1], self.threshold)

        # Compute average factors
        return (forward_factors + backward_factors) / 2

    @staticmethod
    def _compute_factors(tokens: list, threshold: float) -> float:
        factors = 0
        current_segment = []
        unique_words = set()

        for token in tokens:
            current_segment.append(token)
            unique_words.add(token)
            ttr = len(unique_words) / len(current_segment)

            if ttr <= threshold:
                factors += 1
                current_segment = []
                unique_words = set()

        # handle last segment
        if current_segment:
            ttr = len(unique_words) / len(current_segment)
            if ttr <= threshold:
                factors += 1
            else:
                factors += 1 - (ttr - threshold) / (1 - threshold)

        return len(tokens) / factors if factors > 0 else len(tokens)
