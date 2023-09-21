
from collections import defaultdict

def extract_bigrams(txtfile, frqfile):
	bigrams = defaultdict(int)

	for word,freq in zip(txtfile, frqfile):
		word = word.strip().decode("utf-8")
		freq = int(freq)

		if freq < 1000:
			break

		for first,second in zip(word, word[1:]):
			bigram = first + second
			bigrams[bigram] += freq

	bigrams = dict(sorted(((k,v) for k,v in bigrams.items()), key=lambda kv: -kv[1]))
	total = sum(bigrams.values())
	bigrams = dict((k,v/total) for k,v in bigrams.items())
	bigrams = dict((k,v) for k,v in bigrams.items() if v > 1e-4)

	return bigrams


if __name__ == '__main__':
	import sys
	import bz2
	lang = sys.argv[1]
	txtfile = bz2.open(lang + ".txt.bz2")
	frqfile = bz2.open(lang + ".frq.bz2")

	bigrams = extract_bigrams(txtfile, frqfile)
	for bigram,freq in bigrams.items():
		print(bigram, freq)
