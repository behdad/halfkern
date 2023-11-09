import ngrams
import cairo as cr
import cairoft
import uharfbuzz as hb
import numpy as np
from scipy import signal
import math
import functools
import unicodedata
from collections import defaultdict


def gaussian(x, a, b, c):
    return a * math.exp(-((x - b) ** 2) / (2 * c**2))


def kernel(width):
    kernel = list(
        gaussian(x, 1, KERNEL_WIDTH // 2, KERNEL_WIDTH / 4) for x in range(KERNEL_WIDTH)
    )
    s = sum(kernel)
    kernel = np.matrix([x / s for x in kernel], dtype="float32")
    return kernel


FONT_FACE = None
HB_FONT = None

FONT_SIZE = 100

KERNEL_WIDTH = round(0.2 * FONT_SIZE)
if KERNEL_WIDTH % 2 == 0:
    KERNEL_WIDTH += 1
KERNEL = kernel(KERNEL_WIDTH)
BIAS = KERNEL_WIDTH // 2


def blur(surface, *, envelope="sdf", kernel=None):
    if kernel is None:
        kernel = KERNEL

    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()
    data = surface.get_data()

    image = []
    for i in range(height):
        image.append(data[i * stride : i * stride + width])
    image = np.matrix(image, dtype="float")

    if envelope == "sdf":
        import skfmm

        image = 255 - (255 / BIAS) * skfmm.distance(255 - image)
        image = np.maximum(image, np.zeros(image.shape))
    elif envelope == "gaussian":
        image = signal.convolve(image, kernel, mode="same")
        image = image.transpose()
        image = signal.convolve(image, kernel, mode="same")
        image = image.transpose()
    else:
        raise ValueError("Unknown envelope type: " + envelope)

    image = np.matrix(image, dtype="uint8")
    stride = (width + 3) & ~3
    padding = ((0, 0), (0, stride - width))
    data = bytearray(np.pad(image, padding).tobytes())

    blurred = cr.ImageSurface.create_for_data(data, cr.FORMAT_A8, width, height, stride)
    ctx = cr.Context(blurred)

    if envelope == "sdf":
        ctx.set_source_rgba(0, 0, 0, 0.5)
        ctx.set_operator(cr.OPERATOR_DEST_OUT)
        ctx.paint()

    ctx.set_operator(cr.OPERATOR_OVER)
    ctx.set_source_surface(surface, 0, 0)
    ctx.paint()

    return blurred


def create_surface_context(width, height):
    surface = cr.ImageSurface(cr.FORMAT_A8, width, height)
    ctx = cr.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    if FONT_FACE is not None:
        ctx.set_font_face(FONT_FACE)
    ctx.set_font_size(FONT_SIZE)
    return ctx


def create_pdf_surface_context(filename):
    surface = cr.PDFSurface(filename, 0, 0)
    ctx = cr.Context(surface)
    if FONT_FACE is not None:
        ctx.set_font_face(FONT_FACE)
    ctx.set_font_size(FONT_SIZE)
    return ctx


@functools.cache
def create_blurred_surface_for_text_cached(text, *, envelope="sdf"):
    glyph = Glyph(text)
    if surface_sum(glyph.surface) == 0:
        return None
    glyph.surface = blur(glyph.surface, envelope=envelope)
    return glyph


class Glyph:
    def __init__(self, text):
        self.text = text

        measurement_ctx = create_surface_context(1, 1)
        font_extents = measurement_ctx.font_extents()
        ascent = math.ceil(font_extents[0])
        descent = math.ceil(font_extents[1])
        height = ascent + descent

        box = measurement_ctx.text_extents(text)
        width = round(box.width)

        ctx = create_surface_context(width + 2 * BIAS, height + 2 * BIAS)
        self.surface = ctx.get_target()
        self.width = self.surface.get_width()
        self.height = self.surface.get_height()
        self.advance = round(box.x_advance)
        self.origin = (math.ceil(-box.x_bearing) + BIAS, ascent + BIAS)

        ctx.move_to(*self.origin)
        ctx.show_text(text)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_advance(self):
        return self.advance

    def get_stride(self):
        return self.surface.get_stride()

    def get_data(self):
        return self.surface.get_data()


def overlap(l, r, kern=0):
    height = l.get_height()
    assert height == r.get_height()

    l_offset = -(l.advance + l.origin[0]) + r.origin[0] - kern
    width = max(0, min(l.width + l_offset, r.width))

    ctx = create_surface_context(width, height)

    ctx.set_source_surface(r.surface, 0, 0)
    ctx.set_operator(cr.OPERATOR_SOURCE)
    ctx.paint()

    ctx.set_source_surface(l.surface, l_offset, 0)
    ctx.set_operator(cr.OPERATOR_IN)
    ctx.paint()

    return ctx.get_target()


def surface_sum(surface, func=sum):
    data = surface.get_data()

    if func is max:
        return np.max(np.array(data, dtype="float32") ** 2)
    elif func is sum:
        return np.sum(np.array(data, dtype="float32") ** 2)

    # Slower but takes stride into account

    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    s = 0
    for i in range(height):
        for j in range(width):
            s = func(s, data[i * stride + j] ** 2)

    return s


def kern_pair(
    l, r, min_overlap, max_overlap, *, reduce=max, envelope="sdf", blurred=False
):
    old_l_surface = l.surface
    old_r_surface = r.surface
    if not blurred:
        l.surface = blur(l.surface, envelope=envelope)
        r.surface = blur(r.surface, envelope=envelope)

    try:
        kern = 0
        s = surface_sum(overlap(l, r, kern), func=reduce)

        if s < min_overlap:
            for kern in range(-1, -2 * BIAS - 1, -1):
                o = overlap(l, r, kern)
                s = surface_sum(o, func=reduce)
                if s >= min_overlap:
                    break
            else:
                return None, 0
        elif s > max_overlap:
            for kern in range(+1, +2 * BIAS + 1, +1):
                o = overlap(l, r, kern)
                s = surface_sum(o, func=reduce)
                if s <= max_overlap:
                    break
            else:
                return None, 0

    finally:
        l.surface = old_l_surface
        r.surface = old_r_surface

    # Return just half the kern
    return kern // 2 if kern < 0 else kern, s


def showcase_pair(l, r, kern1, kern2):
    height = l.get_height()

    ctx = create_surface_context(
        l.get_advance() + r.get_advance() + 2 * BIAS, height * 3 - 2 * BIAS
    )
    ctx.paint()
    ctx.set_operator(cr.OPERATOR_DEST_OUT)

    ctx.set_source_surface(l.surface, -l.origin[0] + BIAS, 0)
    ctx.paint()
    ctx.set_source_surface(r.surface, l.get_advance() + -r.origin[0] + BIAS, 0)
    ctx.paint()

    ctx.translate(0, height - BIAS)

    ctx.set_source_surface(l.surface, -l.origin[0] + BIAS, 0)
    ctx.paint()
    ctx.set_source_surface(r.surface, l.get_advance() + -r.origin[0] + BIAS + kern1, 0)
    ctx.paint()

    ctx.translate(0, height - BIAS)

    ctx.set_source_surface(l.surface, -l.origin[0] + BIAS, 0)
    ctx.paint()
    ctx.set_source_surface(r.surface, l.get_advance() + -r.origin[0] + BIAS + kern2, 0)
    ctx.paint()

    return ctx.get_target()


CONTEXTS = ("non", "HOH")


def showcase_in_context(ctx, l, r, kern1, kern2):
    font_extents = ctx.font_extents()
    ascent = round(font_extents[0])
    descent = round(font_extents[1])
    height = ascent + descent

    ctx.save()
    for op in ("MEASURE", "CUT"):
        width = 0
        lines = 0
        for context in CONTEXTS:
            textl = context + l
            textr = r + context
            for kern in (0, kern1, kern2):
                if op == "MEASURE":
                    # Measure
                    lines += 1
                    this_width = 0
                    box = ctx.text_extents(textl)
                    this_width += box.x_advance
                    this_width += kern
                    box = ctx.text_extents(textr)
                    this_width += box.x_advance
                    width = max(width, this_width)
                else:
                    # Render
                    ctx.move_to(BIAS, BIAS + ascent)
                    ctx.show_text(textl)
                    x, y = ctx.get_current_point()
                    ctx.move_to(x + kern, y)
                    ctx.show_text(textr)
                    ctx.translate(0, height + BIAS)

        if op == "MEASURE":
            ctx.get_target().set_size(
                round(width) + 2 * BIAS, (height + BIAS) * lines + BIAS
            )
        else:
            ctx.show_page()
    ctx.restore()

    return ctx.get_target()


def create_hb_font(fontfile):
    blob = hb.Blob.from_file_path(fontfile)
    face = hb.Face(blob, 0)
    font = hb.Font(face)
    return font


def actual_kern(l, r, scaled=True):
    buf = hb.Buffer()
    buf.add_str(l)
    buf.guess_segment_properties()
    hb.shape(HB_FONT, buf)
    l_advance = sum(g.x_advance for g in buf.glyph_positions)

    buf = hb.Buffer()
    buf.add_str(r)
    buf.guess_segment_properties()
    hb.shape(HB_FONT, buf)
    r_advance = sum(g.x_advance for g in buf.glyph_positions)

    buf = hb.Buffer()
    buf.add_str(l)
    buf.add_str(r)
    buf.guess_segment_properties()
    hb.shape(HB_FONT, buf)
    combined_advance = sum(g.x_advance for g in buf.glyph_positions)

    kern = combined_advance - (l_advance + r_advance)
    if scaled:
        kern = round(kern * FONT_SIZE / HB_FONT.face.upem)
    return kern


TUNING_CHARS = "lno"


def find_s(*, reduce=max, envelope="sdf"):
    global KERNEL_WIDTH, KERNEL, BIAS
    while True:
        ss = []
        for c in TUNING_CHARS:
            glyph = Glyph(c)
            glyph.surface = blur(glyph.surface, envelope=envelope)
            kern, s = kern_pair(glyph, glyph, 0, 1e10, blurred=True, reduce=reduce)
            ss.append(s)

        min_s = min(ss)
        max_s = max(ss)
        if min_s > max_s / 2:
            break

        KERNEL_WIDTH += 2
        KERNEL = kernel(KERNEL_WIDTH)
        BIAS = KERNEL_WIDTH // 2

    if KERNEL_WIDTH > 2 * FONT_SIZE:
        raise Exception("Failed to find reasonable kernel size.")

    return min_s, max_s


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(
        "python3 kern_pair.py",
        description="Autokern pairs of characters.",
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file")
    parser.add_argument("text", metavar="bigram", nargs="*", help="Pair to kern")
    parser.add_argument(
        "--context",
        metavar="context",
        action="append",
        help="Context texts to show",
    )
    parser.add_argument(
        "--reduce",
        metavar="function",
        type=str,
        help="Function to reduce overlaps: 'sum' or 'max'. Default: sum",
    )
    parser.add_argument(
        "--envelope",
        metavar="type",
        type=str,
        help="Envelope type: 'sdf' or 'gaussian'. Default: sdf",
    )
    parser.add_argument(
        "--dict",
        metavar="textfile",
        type=str,
        nargs="*",
        help="Dictionary file to use for bigrams. Default: None",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        type=str,
        help="Dictionary text encoding. Default: utf-8",
    )
    parser.add_argument(
        "-l",
        "--letters-only",
        action="store_true",
        help="Only list bigrams of letters. Default: False",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Tolerance for showing bigram kerning value. Default: 0.033",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Bigram cutoff probability if dictionary is provided. Default: .999",
    )

    options = parser.parse_args(sys.argv[1:])

    if not options.text and not options.dict:
        parser.print_help()
        sys.exit(1)

    font = options.font
    texts = options.text
    if options.context:
        CONTEXTS = options.context

    import builtins

    reduce = getattr(builtins, options.reduce or "sum")
    assert reduce in {max, sum}
    envelope = options.envelope or "sdf"

    FONT_FACE = cairoft.create_cairo_font_face_for_file(font, 0)
    HB_FONT = create_hb_font(font)

    if len(texts) == 1 and len(texts[0]) == 1:
        _, _ = find_s(reduce=reduce, envelope=envelope)
        glyph = Glyph(texts[0])
        glyph.surface = blur(glyph.surface, envelope=envelope)
        glyph.surface.write_to_png("envelope.png")
        sys.exit(0)

    assert all(len(text) == 2 for text in texts)

    min_s, max_s = find_s(reduce=reduce, envelope=envelope)

    pdf_ctx = create_pdf_surface_context("kerned.pdf")

    # Process individual pairs

    for text in texts:
        l = Glyph(text[0])
        r = Glyph(text[1])

        kern, s = kern_pair(l, r, min_s, max_s, reduce=reduce, envelope=envelope)
        if kern is None:
            print("Couldn't autokern")
            kern = 0
        font_kern = actual_kern(text[0], text[1])
        font_kern_upem = actual_kern(text[0], text[1], scaled=False)

        upem = HB_FONT.face.upem
        print(
            text,
            "autokern:",
            kern,
            "(%u units)" % round(kern / FONT_SIZE * upem),
            "existing kern:",
            font_kern,
            "(%u units)" % round(font_kern_upem),
        )
        print("Saving kern.png")
        s = showcase_pair(l, r, kern, font_kern)
        s.write_to_png("kern.png")
        s = showcase_in_context(pdf_ctx, text[0], text[1], kern, font_kern)
        s.write_to_png("kerned.png")

    # Process dictionaries

    encoding = options.encoding or "utf-8"
    tolerance = options.tolerance or 0.033
    if tolerance >= 1:
        tolerance = tolerance / kern.FONT_SIZE
    cutoff = options.cutoff or 0.999
    if cutoff >= 1:
        cutoff = cutoff / 100.0

    ngrams.ENCODING = encoding
    ngrams.LETTERS_ONLY = options.letters_only

    all_bigrams = defaultdict(int)
    for dictfile in options.dict or []:
        this_bigrams = ngrams.extract_ngrams_from_file(2, dictfile, cutoff=cutoff)
        for k, v in this_bigrams.items():
            all_bigrams[k] += v
    for bigram in all_bigrams:
        if (
            unicodedata.category(bigram[0]) == "Mn"
            or unicodedata.category(bigram[1]) == "Mn"
        ):
            continue

        l = create_blurred_surface_for_text_cached(bigram[0], envelope=envelope)
        r = create_blurred_surface_for_text_cached(bigram[1], envelope=envelope)

        if l is None or r is None:
            continue

        kern_value, _ = kern_pair(l, r, min_s, max_s, blurred=True, reduce=reduce)
        if kern_value is None:
            continue
        font_kern = actual_kern(bigram[0], bigram[1])
        if kern_value == 0 and font_kern == 0:
            continue

        if abs(kern_value - font_kern) <= FONT_SIZE * tolerance:
            continue

        showcase_in_context(pdf_ctx, *bigram, kern_value, font_kern)

        print(bigram, kern_value, font_kern)
