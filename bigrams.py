from collections import defaultdict
import itertools

MIN_FREQ = 10
ENCODING = "utf-8"


def extract_bigrams(txtfile, frqfile):
    bigrams = defaultdict(int)

    for word, freq in zip(txtfile, frqfile):
        try:
            word = word.strip().decode(ENCODING)
        except UnicodeDecodeError:
            continue
        freq = int(freq)

        if freq < MIN_FREQ:
            break

        for first, second in zip(word, word[1:]):
            if first in "0123456789" or second in "0123456789":
                continue
            bigram = first + second
            bigrams[bigram] += freq

    bigrams = dict(sorted(((k, v) for k, v in bigrams.items()), key=lambda kv: -kv[1]))
    total = sum(bigrams.values())
    bigrams = dict((k, v / total) for k, v in bigrams.items())
    cutoff = iter(bigrams.values()).__next__() * 1e-6
    bigrams = dict((k, v) for k, v in bigrams.items() if v > cutoff)

    return bigrams


def extract_bigrams_from_file(filename):
    try:
        txtfile = open(filename, "rb")
        # Assume hunspell dictionary format; drop everything after "/"
        txtfile = (s if s.find(b"/") == -1 else s[: s.find(b"/")] for s in txtfile)
        frqfile = itertools.cycle([MIN_FREQ])
    except FileNotFoundError:
        import bz2

        # Assume harfbuzz-testing-wikipedia format
        txtfile = bz2.open(filename + ".txt.bz2")
        frqfile = bz2.open(filename + ".frq.bz2")

    return extract_bigrams(txtfile, frqfile)


if __name__ == "__main__":
    import sys

    lang = sys.argv[1]
    import argparse

    parser = argparse.ArgumentParser(
        "python3 bigrams.py",
        description="Find bigrams from a language dictionary.",
    )
    parser.add_argument("dict", metavar="dict", help="Dictionary file.")
    parser.add_argument(
        "-e",
        "--encoding",
        type=str,
        help="Text encoding. Default: utf-8",
    )

    options = parser.parse_args(sys.argv[1:])

    dictfile = options.dict
    encoding = options.encoding or "utf-8"

    ENCODING = encoding

    bigrams = extract_bigrams_from_file(dictfile)

    for bigram, freq in bigrams.items():
        print(bigram, freq)
