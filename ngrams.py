from collections import defaultdict
import itertools

MIN_FREQ = 10
ENCODING = "utf-8"
LETTERS_ONLY = False


def extract_ngrams(n, txtfile, frqfile = None):
    if frqfile is None:
        frqfile = itertools.cycle([MIN_FREQ])

    ngrams = defaultdict(int)

    for word, freq in zip(txtfile, frqfile):
        if freq < MIN_FREQ:
            continue

        try:
            word = word.strip().decode(ENCODING)
        except UnicodeDecodeError:
            continue
        freq = int(freq)

        if len(word) < n:
            continue

        words = [word[i:] for i in range(n)]

        for ngram in zip(*words):
            if LETTERS_ONLY and any(not c.isalpha() for c in ngram):
                continue

            ngram = "".join(ngram)
            ngrams[ngram] += freq

    ngrams = dict(sorted(((k, v) for k, v in ngrams.items()), key=lambda kv: -kv[1]))
    total = sum(ngrams.values())
    ngrams = dict((k, v / total) for k, v in ngrams.items())
    cutoff = next(iter(ngrams.values())) * 1e-6
    ngrams = dict((k, v) for k, v in ngrams.items() if v > cutoff)

    return ngrams


def extract_ngrams_from_file(n, filename):
    try:
        txtfile = open(filename, "rb")
        # Assume hunspell dictionary format; drop everything after "/"
        txtfile = (s if s.find(b"/") == -1 else s[: s.find(b"/")] for s in txtfile)
        frqfile = None
    except FileNotFoundError:
        import bz2

        # Assume harfbuzz-testing-wikipedia format
        txtfile = bz2.open(filename + ".txt.bz2")
        frqfile = bz2.open(filename + ".frq.bz2")

    return extract_ngrams(n, txtfile, frqfile)


if __name__ == "__main__":
    import sys

    lang = sys.argv[1]
    import argparse

    parser = argparse.ArgumentParser(
        "python3 ngrams.py",
        description="Find ngrams from a language dictionary.",
    )
    parser.add_argument("dict", metavar="dict", help="Dictionary file.")
    parser.add_argument(
        "-n",
        "--ngram",
        type=int,
        help="Length of ngrams. Default: 2",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=str,
        help="Text encoding. Default: utf-8",
    )
    parser.add_argument(
        "-l",
        "--letters-only",
        action="store_true",
        help="Only list ngrams of letters. Default: False",
    )

    options = parser.parse_args(sys.argv[1:])

    dictfile = options.dict
    encoding = options.encoding or "utf-8"
    ngram = options.ngram or 2

    ENCODING = encoding
    LETTERS_ONLY = options.letters_only

    ngrams = extract_ngrams_from_file(ngram, dictfile)

    for ngram, freq in ngrams.items():
        print(ngram, freq)
