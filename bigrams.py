from collections import defaultdict

MIN_FREQ = 10


def extract_bigrams(txtfile, frqfile):
    bigrams = defaultdict(int)

    for word, freq in zip(txtfile, frqfile):
        try:
            word = word.strip().decode("utf-8")
        except UnicodeDecodeError:
            word = word.strip().decode("latin1")
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


if __name__ == "__main__":
    import sys
    import bz2

    lang = sys.argv[1]
    txtfile = bz2.open(lang + ".txt.bz2")
    frqfile = bz2.open(lang + ".frq.bz2")

    bigrams = extract_bigrams(txtfile, frqfile)
    for bigram, freq in bigrams.items():
        print(bigram, freq)
