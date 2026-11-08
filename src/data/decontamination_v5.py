"""Read-only training-vs-evaluation overlap audit, version 5.

Four independent lexical rules. A training row is flagged if ANY rule fires:

1. ``normalized_exact``: identical word sequence after NFKC + casefold.
2. ``jaccard_3gram``: the repository's original rule (word 3-gram Jaccard >= 0.85,
   ``src/data/prepare_math_data.SimilarityIndex``).
3. ``ngram13_lcs``: Qwen2.5-Math style. The two texts share at least one word
   13-gram AND their longest common (word) subsequence covers >= 60% of the
   shorter text.
4. ``short_containment``: a protected text shorter than 13 words appears as a
   contiguous word sequence inside the training text and makes up at least half
   of it (13-grams cannot see such short items).

Lexical rules cannot prove semantic or pretraining decontamination; they only
make the overlap policy explicit and reproducible.
"""
import ast
import hashlib
import re
import unicodedata
from collections import defaultdict

from src.data.prepare_math_data import SimilarityIndex

NGRAM = 13
LCS_RATIO = 0.6
SHORT_CONTAINMENT_RATIO = 0.5
RULES = ('normalized_exact', 'jaccard_3gram', 'ngram13_lcs', 'short_containment')


# Fixed wrapper sentences of open-r1/verifiable-coding-problems-python and of
# the project's interface suffixes. They are identical across unrelated tasks,
# so they are removed before overlap scoring (problem content is untouched).
CODE_BOILERPLATE = (
    r'solve the following coding problem using the programming language python\s*:',
    r'the input will be stdin and you should print your solution to stdout',
    r'the input will be given via stdin and the output should be printed to stdout by your code\.?',
    r'now solve the problem (?:and return the code|by providing the code)\.?',
    r'write a complete python program that reads standard input and writes standard output\.?',
    r'required python interface \(return the result; do not read stdin\):',
    r'-{3,}\s*(?:input|output|examples?|note|notes)\s*-{3,}',
    # AceCode task opener shared by thousands of unrelated generated tasks.
    r'you are tasked with (?:creating|implementing|writing|developing) an? (?:simple )?(?:python )?'
    r'(?:function|program|method|script)(?: that| which)?',
)
_BOILERPLATE = re.compile('|'.join(CODE_BOILERPLATE), re.I)


def strip_code_boilerplate(text):
    return _BOILERPLATE.sub(' ', text)


def words(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = text.replace('\\left', ' ').replace('\\right', ' ')
    # Punctuation and TeX symbols are dropped; TeX command names stay as words.
    return re.findall(r'\w+', text)


def ngrams(tokens, n=NGRAM):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def lcs_length_dp(a, b):
    """Reference O(len(a) * len(b)) dynamic program (used by tests)."""
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def lcs_length(a, b):
    """Exact LCS length with the bit-parallel recurrence (Allison-Dix / Hyyro).

    Bits of ``V`` index positions of ``b``; after processing ``a`` the number of
    zero bits equals the LCS length. Python integers give arbitrary width.
    """
    if not a or not b:
        return 0
    match = {}
    for i, token in enumerate(b):
        match[token] = match.get(token, 0) | (1 << i)
    mask = (1 << len(b)) - 1
    v = mask
    for token in a:
        u = v & match.get(token, 0)
        v = ((v + u) | (v - u)) & mask
    return len(b) - bin(v).count('1')


def lcs_ratio(a, b):
    shorter = min(len(a), len(b))
    return lcs_length(a, b) / shorter if shorter else 0.


def ast_sha(code):
    """Same definition as reference_ast_sha256 in the v2-v4 releases."""
    return hashlib.sha256(ast.dump(ast.parse(code), include_attributes=False).encode()).hexdigest()


class ProtectedIndex:
    """Index of protected (evaluation/dev) texts. ``items``: dicts with id, text, set."""

    def __init__(self, items, jaccard_threshold=.85):
        self.items = list(items)
        if len({i['id'] for i in self.items}) != len(self.items):
            raise ValueError('Duplicate protected IDs')
        self.tokens = [words(i['text']) for i in self.items]
        self.exact = defaultdict(list)
        self.postings = defaultdict(set)
        self.short = defaultdict(list)
        self.short_lengths = set()
        self.jaccard = SimilarityIndex(jaccard_threshold)
        for idx, (item, toks) in enumerate(zip(self.items, self.tokens)):
            self.exact[tuple(toks)].append(idx)
            for gram in ngrams(toks):
                self.postings[gram].add(idx)
            if 0 < len(toks) < NGRAM:
                self.short[tuple(toks)].append(idx)
                self.short_lengths.add(len(toks))
            self.jaccard.add({'sample_id': str(idx), 'problem': item['text']})

    def _hit(self, idx, rule, **evidence):
        item = self.items[idx]
        return {'protected_id': item['id'], 'protected_set': item['set'], 'rule': rule, **evidence}

    def query(self, text):
        toks = words(text)
        hits = {}
        for idx in self.exact.get(tuple(toks), ()):
            hits.setdefault(idx, self._hit(idx, 'normalized_exact', similarity=1.))
        match = self.jaccard.match(text)
        if match is not None:
            idx = int(match['matched_id'])
            hits.setdefault(idx, self._hit(idx, 'jaccard_3gram', similarity=round(match['similarity'], 4),
                                           kind=match['kind']))
        candidates = set()
        for gram in ngrams(toks):
            candidates.update(self.postings.get(gram, ()))
        for idx in sorted(candidates - set(hits)):
            ratio = lcs_ratio(toks, self.tokens[idx])
            if ratio >= LCS_RATIO:
                hits[idx] = self._hit(idx, 'ngram13_lcs', lcs_ratio=round(ratio, 4))
        for n in sorted(self.short_lengths):
            if n > len(toks) or n / len(toks) < SHORT_CONTAINMENT_RATIO:
                continue
            for i in range(len(toks) - n + 1):
                for idx in self.short.get(tuple(toks[i:i + n]), ()):
                    hits.setdefault(idx, self._hit(idx, 'short_containment', length_ratio=round(n / len(toks), 4)))
        return [hits[i] for i in sorted(hits)]


def audit(train_rows, index, text_of):
    """Return {sample_id: [hits]} for flagged rows only."""
    flagged = {}
    for row in train_rows:
        hits = index.query(text_of(row))
        if hits:
            flagged[row['sample_id']] = hits
    return flagged
