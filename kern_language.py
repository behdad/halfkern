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


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(
        "python3 kern_language.py",
        description="Find missing kerning pairs for a font & language pair.",
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file.")
    parser.add_argument("dict", metavar="dict", help="Dictionary file.")
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

    options = parser.parse_args(sys.argv[1:])

    fontfile = options.font
    dictfile = options.dict
    encoding = options.encoding or "utf-8"

    bigrams.ENCODING = encoding
    bigrams.LETTERS_ONLY = options.letters_only
    kern.FONT_FACE = cairoft.create_cairo_font_face_for_file(fontfile, 0)
    kern.HB_FONT = kern.create_hb_font(fontfile)

    s = kern.find_s()

    all_bigrams = bigrams.extract_bigrams_from_file(dictfile)
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

        kern_value, _ = kern.kern_pair(l, r, s, blurred=True)
        if kern_value is None:
            continue
        font_kern = kern.actual_kern(bigram[0], bigram[1])
        if kern_value == 0 and font_kern == 0:
            continue

        if abs(kern_value - font_kern) <= kern.FONT_SIZE / 32:
            continue

        # if kern_value * 2 <= font_kern <= kern_value:
        #    continue

        print(bigram, kern_value, font_kern)
