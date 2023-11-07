import ngrams
import kern_pair as kern
import cairoft
import functools
import unicodedata
from collections import defaultdict


@functools.cache
def create_blurred_surface_for_text(text):
    glyph = kern.Glyph(text)
    if kern.surface_sum(glyph.surface) == 0:
        return None
    glyph.surface = kern.blur(glyph.surface)
    return glyph


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(
        "python3 kern_pairs.py",
        description="Find missing kerning pairs for a font & language pair.",
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file.")
    parser.add_argument("dict", metavar="dict", nargs="+", help="Dictionary file.")
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
        help="Only list bigrams of letters. Default: False",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        help="Tolerance for kerning value. Default: 0.033.",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        help="Cutoff probability. Default: .999",
    )

    options = parser.parse_args(sys.argv[1:])

    fontfile = options.font
    dictfiles = options.dict
    encoding = options.encoding or "utf-8"
    tolerance = options.tolerance or 0.033
    if tolerance >= 1:
        tolerance = tolerance / kern.FONT_SIZE
    cutoff = options.cutoff or 0.999
    if cutoff >= 1:
        cutoff = cutoff / 100.0

    ngrams.ENCODING = encoding
    ngrams.LETTERS_ONLY = options.letters_only
    kern.FONT_FACE = cairoft.create_cairo_font_face_for_file(fontfile, 0)
    kern.HB_FONT = kern.create_hb_font(fontfile)

    min_s, max_s = kern.find_s()

    all_bigrams = defaultdict(int)
    for dictfile in dictfiles:
        this_bigrams = ngrams.extract_ngrams_from_file(2, dictfile, cutoff=cutoff)
        for k, v in this_bigrams.items():
            all_bigrams[k] += v
    for bigram in all_bigrams:
        if (
            unicodedata.category(bigram[0]) == "Mn"
            or unicodedata.category(bigram[1]) == "Mn"
        ):
            continue

        l = create_blurred_surface_for_text(bigram[0])
        r = create_blurred_surface_for_text(bigram[1])

        if l is None or r is None:
            continue

        kern_value, _ = kern.kern_pair(l, r, min_s, max_s, blurred=True)
        if kern_value is None:
            continue
        font_kern = kern.actual_kern(bigram[0], bigram[1])
        if kern_value == 0 and font_kern == 0:
            continue

        if abs(kern_value - font_kern) <= kern.FONT_SIZE * tolerance:
            continue

        print(bigram, kern_value, font_kern)
