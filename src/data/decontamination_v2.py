"""Read-only exact, token-5-gram and numeric-template nearest-neighbor audit."""
import collections
import hashlib
import re
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def tokens(text, numeric=False):
    words = re.findall(r'\w+|[^\w\s]', normalize(text))
    return ['<num>' if numeric and re.fullmatch(r'\d+(?:\.\d+)?', x) else x for x in words]


def grams(text, numeric=False):
    values = tokens(text, numeric)
    return {tuple(values[i:i+5]) for i in range(len(values)-4)}


def similarity(a, b):
    return len(a & b) / len(a | b) if a and b else 0.


def containment(a, b):
    # Detect short wrappers; require >=8 shingles and bounded length ratio so
    # a tiny shared phrase cannot classify a long problem as a duplicate.
    if not a or not b or min(len(a), len(b)) < 8 or min(len(a), len(b)) / max(len(a), len(b)) < .5:
        return 0.
    return len(a & b) / min(len(a), len(b))


class Index:
    def __init__(self, rows):
        self.rows = rows
        self.exact = collections.defaultdict(list)
        self.standard, self.numeric = [], []
        self.postings = collections.defaultdict(set)
        for i, row in enumerate(rows):
            text = row['text']
            self.exact[normalize(text)].append(i)
            a, b = grams(text), grams(text, True)
            self.standard.append(a)
            self.numeric.append(b)
            for g in a | b:
                self.postings[g].add(i)

    def query(self, text, threshold):
        a, b = grams(text), grams(text, True)
        candidates = set(self.exact.get(normalize(text), []))
        for g in a | b:
            candidates.update(self.postings.get(g, ()))
        hits = []
        for i in sorted(candidates):
            exact = normalize(text) == normalize(self.rows[i]['text'])
            raw = similarity(a, self.standard[i])
            numeric = similarity(b, self.numeric[i])
            contained = max(containment(a, self.standard[i]), containment(b, self.numeric[i]))
            if exact or max(raw, numeric, contained) >= threshold:
                hits.append({'target_id': self.rows[i]['id'], 'exact': exact,
                             'jaccard_5gram': raw, 'numeric_template_5gram': numeric,
                             'containment_5gram': contained,
                             'status': 'candidate_requires_review'})
        return hits


def calibrate(positive_pairs, negative_pairs):
    if not positive_pairs:
        raise ValueError('No known positive calibration pairs')
    def score(pair):
        a, b = pair
        return 1. if normalize(a) == normalize(b) else max(similarity(grams(a), grams(b)), similarity(grams(a, True), grams(b, True)), containment(grams(a), grams(b)), containment(grams(a, True), grams(b, True)))
    pos = [score(x) for x in positive_pairs]
    neg = [score(x) for x in negative_pairs]
    candidates = [.5, .6, .7, .75, .8, .85, .9, .95, 1.]
    # Highest pre-scan threshold attaining >=95% known-positive recall.
    accepted = [t for t in candidates if sum(s >= t for s in pos) / len(pos) >= .95]
    threshold = max(accepted) if accepted else min(candidates)
    return {'threshold': threshold, 'target_recall': .95, 'target_reached': bool(accepted),
            'selection': ('highest grid threshold reaching >=0.95 known-positive recall; chosen before train scan' if accepted else 'No grid threshold reached 0.95 recall; use lowest grid 0.5 as unvalidated screening threshold before scan'),
            'known_positive_count': len(pos), 'positive_recall': sum(s >= threshold for s in pos) / len(pos),
            'negative_count': len(neg), 'negative_false_positive_rate': sum(s >= threshold for s in neg) / len(neg) if neg else None,
            'positive_scores': pos, 'negative_scores': neg,
            'limitation': 'Known MBPP identity overlaps do not establish recall for free paraphrases or cross-domain math'}
