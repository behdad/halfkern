import bigrams
import kern
import functools

@functools.cache
def create_surface_for_text(text):
    return kern.create_surface_for_text(text)

if __name__ == '__main__':
    import sys
    import bz2
    lang = sys.argv[1]
    txtfile = bz2.open(lang + ".txt.bz2")
    frqfile = bz2.open(lang + ".frq.bz2")

    all_bigrams = bigrams.extract_bigrams(txtfile, frqfile)
    for bigram in all_bigrams:
        l = create_surface_for_text(bigram[0])
        r = create_surface_for_text(bigram[1])

        kern_value = kern.kern_pair(l, r)
        if kern_value == 0:
            continue

        print(kern_value, bigram)
