from collections import defaultdict
import itertools

MIN_FREQ = 10
LETTERS_ONLY = False


def extract_ngrams(
    text, n, *, frequencies=None, cutoff=0.999, min_freq=MIN_FREQ, encoding="utf-8"
):
    if frequencies is None:
        frequencies = itertools.cycle([min_freq])

    ngrams = defaultdict(int)

    for word, freq in zip(text, frequencies):
        freq = int(freq)
        if freq < min_freq:
            continue

        try:
            word = word.strip().decode(encoding)
        except UnicodeDecodeError:
            continue
        freq = int(freq)

        if len(word) < n:
            continue

        words = [word[i:] for i in range(n)]

        for ngram in zip(*words):
            ngram = "".join(ngram)
            ngrams[ngram] += freq

    if LETTERS_ONLY:
        for ngram in list(ngrams.keys()):
            if any(not c.isalpha() for c in ngram):
                del ngrams[ngram]

    ngrams = dict(sorted(((k, v) for k, v in ngrams.items()), key=lambda kv: -kv[1]))
    total = sum(ngrams.values())
    ngrams = dict((k, v / total) for k, v in ngrams.items())
    total = 0
    new_ngrams = {}
    for k, v in ngrams.items():
        total += v
        if total > cutoff:
            break
        new_ngrams[k] = v

    return new_ngrams


def extract_ngrams_from_file(filename, *kargs, **kwargs):
    frqfile = None
    try:
        txtfile = open(filename, "rb")
    except FileNotFoundError:
        try:
            import bz2

            # Assume harfbuzz-testing-wikipedia format
            txtfile = bz2.open(filename + ".txt.bz2").read().splitlines()
            frqfile = bz2.open(filename + ".frq.bz2").read().splitlines()
        except FileNotFoundError:
            try:
                # Assume hunspell dictionary format;
                afffile = open(filename + ".aff", "rb")
                for line in afffile:
                    if line.startswith(b"SET"):
                        kwargs["encoding"] = (
                            line.replace(b"\t", b" ").split()[1].decode("ascii")
                        )
                        break
                txtfile = open(filename + ".dic", "rb")
                next(txtfile)  # Skip over the num entries line
                txtfile = (
                    s if s.find(b"/") == -1 else s[: s.find(b"/")] for s in txtfile
                )

            except FileNotFoundError:
                raise FileNotFoundError("File not found: %s" % filename)

    return extract_ngrams(txtfile, *kargs, frequencies=frqfile, **kwargs)


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
        "-c",
        "--cutoff",
        type=float,
        help="Cutoff probability. Default: .999",
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
    cutoff = options.cutoff or 0.999

    LETTERS_ONLY = options.letters_only

    ngrams = extract_ngrams_from_file(ngram, dictfile, cutoff=cutoff, encoding=encoding)

    for ngram, freq in ngrams.items():
        print(ngram, freq)
