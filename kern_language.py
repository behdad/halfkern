import bigrams
import kern
import cairoft
import functools
import unicodedata

@functools.cache
def create_blurred_surface_for_text(text):
    surface = kern.create_surface_for_text(text)
    if kern.surface_sum(surface) == 0:
        return None
    return kern.blur(surface, kern.KERNEL)

if __name__ == '__main__':
    import sys
    font = sys.argv[1]
    lang = sys.argv[2]

    kern.FONT_FACE = cairoft.create_cairo_font_face_for_file(font, 0)
    kern.HB_FONT = kern.create_hb_font(font)

    l = create_blurred_surface_for_text('l')
    _, sl = kern.kern_pair(l, l, 0, blurred=True)
    l = create_blurred_surface_for_text('n')
    _, sn = kern.kern_pair(l, l, 0, blurred=True)
    l = create_blurred_surface_for_text('o')
    _, so = kern.kern_pair(l, l, 0, blurred=True)
    s = min(sl, sn, so)

    import bz2
    txtfile = bz2.open(lang + ".txt.bz2")
    frqfile = bz2.open(lang + ".frq.bz2")

    all_bigrams = bigrams.extract_bigrams(txtfile, frqfile)
    for bigram in all_bigrams:

        if unicodedata.category(bigram[0]) == 'Mn' or unicodedata.category(bigram[1]) == 'Mn':
            continue

        l = create_blurred_surface_for_text(bigram[0])
        r = create_blurred_surface_for_text(bigram[1])

        if l is None or r is None:
            continue

        kern_value, _ = kern.kern_pair(l, r, s, blurred=True)
        font_kern = kern.actual_kern(bigram[0], bigram[1])
        if kern_value == 0 and font_kern == 0:
            continue

        if abs(kern_value - font_kern) <= 2:
            continue

        print(bigram, kern_value, font_kern)
