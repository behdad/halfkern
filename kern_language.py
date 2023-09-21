import bigrams
import kern
import cairoft
import functools

@functools.cache
def create_blurred_surface_for_text(text):
    return kern.blur(kern.create_surface_for_text(text), kern.KERNEL)

if __name__ == '__main__':
    import sys
    font = sys.argv[1]
    lang = sys.argv[2]

    kern.FONT_FACE = cairoft.create_cairo_font_face_for_file(font, 0)

    import bz2
    txtfile = bz2.open(lang + ".txt.bz2")
    frqfile = bz2.open(lang + ".frq.bz2")

    all_bigrams = bigrams.extract_bigrams(txtfile, frqfile)
    for bigram in all_bigrams:
        l = create_blurred_surface_for_text(bigram[0])
        r = create_blurred_surface_for_text(bigram[1])

        kern_value = kern.kern_pair(l, r, blurred=True)
        if kern_value == 0:
            continue

        print(kern_value, bigram)
