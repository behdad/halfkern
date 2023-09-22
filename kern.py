import cairo as cr
import cairoft
import uharfbuzz as hb
import numpy as np
from scipy import signal
import math


def gaussian(x, a, b, c):
    return a * math.exp(-((x - b) ** 2) / (2 * c**2))


def kernel(width):
    kernel = list(
        gaussian(x, 1, KERNEL_WIDTH // 2, KERNEL_WIDTH / 4) for x in range(KERNEL_WIDTH)
    )
    s = sum(kernel)
    kernel = np.matrix([x / s for x in kernel])
    kernel = kernel.transpose() * kernel
    return kernel


FONT_FACE = None
HB_FONT = None

FONT_SIZE = 64

KERNEL_WIDTH = round(FONT_SIZE * 0.2)
if KERNEL_WIDTH % 2 == 0:
    KERNEL_WIDTH += 1
KERNEL = kernel(KERNEL_WIDTH)
BIAS = len(KERNEL) // 2


def blur(surface, kernel=None):
    if kernel is None:
        kernel = KERNEL

    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()
    data = surface.get_data()

    image = []
    for i in range(height):
        image.append(data[i * stride : i * stride + width])
    image = np.matrix(image, dtype="uint8")

    image = signal.convolve2d(image, kernel, mode="same")

    image = np.matrix(image, dtype="uint8")
    stride = (width + 3) & ~3
    padding = b"\0" * (stride - width)
    data = bytearray()
    for i in range(height):
        data.extend(image[i].tobytes())
        data.extend(padding)

    blurred = cr.ImageSurface.create_for_data(data, cr.FORMAT_A8, width, height, stride)
    ctx = cr.Context(blurred)
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


def create_surface_for_text(text):
    measurement_ctx = create_surface_context(1, 1)
    font_extents = measurement_ctx.font_extents()
    ascent = font_extents[0]
    descent = font_extents[1]
    height = ascent + descent

    box = measurement_ctx.text_extents(text)
    width = box.x_advance

    ctx = create_surface_context(round(width) + 2 * BIAS, round(height) + 2 * BIAS)

    ctx.move_to(BIAS, BIAS + round(ascent))
    ctx.show_text(text)

    return ctx.get_target()


def overlap(l, r, kern=0):
    height = l.get_height()
    assert height == r.get_height()

    width = r.get_width() + kern

    ctx = create_surface_context(width, height)

    ctx.set_source_surface(r, 0, 0)
    ctx.set_operator(cr.OPERATOR_SOURCE)
    ctx.paint()

    ctx.set_source_surface(l, -l.get_width() + 2 * BIAS - kern, 0)
    ctx.set_operator(cr.OPERATOR_IN)
    ctx.paint()

    return ctx.get_target()


def surface_sum(surface):
    data = surface.get_data()

    return max(np.array(data, dtype="uint8"))

    # Slower but takes stride into account

    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    s = 0
    for i in range(height):
        for j in range(width):
            s = max(s, data[i * stride + j])

    return s


def kern_pair(l, r, min_overlap, max_overlap, blurred=False):
    if not blurred:
        l = blur(l)
        r = blur(r)

    kern = 0
    s = surface_sum(overlap(l, r, kern))

    if s < min_overlap:
        for kern in range(-1, -2 * BIAS - 1, -1):
            o = overlap(l, r, kern)
            s = surface_sum(o)
            if s >= min_overlap:
                break
    elif s > max_overlap:
        for kern in range(+1, +2 * BIAS + 1, +1):
            o = overlap(l, r, kern)
            s = surface_sum(o)
            if s <= max_overlap:
                break

    if s == 0:
        # No overlap
        return None, 0

    # Return just half the kern
    return kern // 2, s


def showcase(l, r, kern1, kern2):
    height = l.get_height()

    ctx = create_surface_context(
        l.get_width() + r.get_width() - 2 * BIAS, height * 3 - 2 * BIAS
    )
    ctx.paint()
    ctx.set_operator(cr.OPERATOR_DEST_OUT)

    ctx.set_source_surface(l, 0, 0)
    ctx.paint()
    ctx.set_source_surface(r, l.get_width() - 2 * BIAS, 0)
    ctx.paint()

    ctx.translate(0, height - BIAS)

    ctx.set_source_surface(l, 0, 0)
    ctx.paint()
    ctx.set_source_surface(r, l.get_width() - 2 * BIAS + kern1, 0)
    ctx.paint()

    ctx.translate(0, height - BIAS)

    ctx.set_source_surface(l, 0, 0)
    ctx.paint()
    ctx.set_source_surface(r, l.get_width() - 2 * BIAS + kern2, 0)
    ctx.paint()

    return ctx.get_target()


CONTEXTS = ("non", "HOH")


def showcase_in_context(l, r, kern1, kern2):
    measurement_ctx = create_surface_context(1, 1)
    font_extents = measurement_ctx.font_extents()
    ascent = round(font_extents[0])
    descent = round(font_extents[1])
    height = ascent + descent

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
                    box = measurement_ctx.text_extents(textl)
                    this_width += box.x_advance
                    this_width += kern
                    box = measurement_ctx.text_extents(textr)
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
            ctx = create_surface_context(
                round(width) + 2 * BIAS, (height + BIAS) * lines + BIAS
            )
            ctx.paint()
            ctx.set_operator(cr.OPERATOR_DEST_OUT)

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


def find_s():
    global KERNEL_WIDTH, KERNEL, BIAS
    while True:
        ss = []
        for c in TUNING_CHARS:
            surface = blur(create_surface_for_text(c))
            kern, s = kern_pair(surface, surface, 0, 1e10, blurred=True)
            ss.append(s)

        min_s = min(ss)
        max_s = max(ss)
        if min_s > max_s / 2:
            return min_s, max_s

        KERNEL_WIDTH += 2
        KERNEL = kernel(KERNEL_WIDTH)
        BIAS = len(KERNEL) // 2


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(
        "python3 kern.py",
        description="Autokern a pair of characters.",
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file.")
    parser.add_argument("text", metavar="text", help="Pair to kern.")
    parser.add_argument(
        "-c",
        "--context",
        metavar="context",
        action="append",
        help="Context texts to show.",
    )

    options = parser.parse_args(sys.argv[1:])

    font = options.font
    text = options.text
    if options.context:
        CONTEXTS = options.context

    FONT_FACE = cairoft.create_cairo_font_face_for_file(font, 0)
    HB_FONT = create_hb_font(font)

    if len(text) == 1:
        surface = create_surface_for_text(text)
        surface = blur(surface, KERNEL)
        surface.write_to_png("kern.png")
        sys.exit(0)

    assert len(text) == 2

    min_s, max_s = find_s()

    l = create_surface_for_text(text[0])
    r = create_surface_for_text(text[1])

    kern, s = kern_pair(l, r, min_s, max_s)
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
    print("Saving kern.png and kerned.png")
    s = showcase(l, r, kern, font_kern)
    s.write_to_png("kern.png")
    s = showcase_in_context(text[0], text[1], kern, font_kern)
    s.write_to_png("kerned.png")
