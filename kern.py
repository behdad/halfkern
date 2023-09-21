import cairo as cr
import cairoft
import uharfbuzz as hb
import math

def gaussian(x, a, b, c):
    return a * math.exp(-(x - b)**2 / (2 * c**2))

KERNEL_WIDTH = 13
KERNEL = list(gaussian(x, 1, KERNEL_WIDTH // 2, KERNEL_WIDTH / 4) for x in range(KERNEL_WIDTH))
BIAS = len(KERNEL) // 2

FONT_FACE = None
FONT_SIZE = 64

HB_FONT = None

def blur(surface, kernel):
    s = sum(kernel)
    kernel = [x/s for x in kernel]

    data = surface.get_data()
    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    s1 = cr.ImageSurface(cr.FORMAT_A8, width, height)
    d1 = s1.get_data()
    stride1 = s1.get_stride()
    for i in range(height):
        for j in range(width):
            p = 0
            for k,v in enumerate(kernel):
                if j + k - BIAS >= 0 and j + k - BIAS < width:
                    p += data[i*stride + j + k - BIAS] * v
            d1[i*stride1 + j] = int(p)

    s2 = cr.ImageSurface(cr.FORMAT_A8, width, height)
    d2 = s2.get_data()
    stride2 = s2.get_stride()
    for j in range(width):
        for i in range(height):
            p = 0
            for k,v in enumerate(kernel):
                if i + k - BIAS >= 0 and i + k - BIAS < height:
                    p += d1[(i + k - BIAS) * stride1 + j] * v
            d2[i*stride2 + j] = int(p)

    s2.mark_dirty()

    return s2

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
    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    s = 0
    for i in range(height):
        for j in range(width):
            s += data[i*stride + j]

    return s

def kern_pair(l, r, min_overlap, blurred=False):

    if not blurred:
        l = blur(l, KERNEL)
        r = blur(r, KERNEL)

    for kern in range(0, -2 * BIAS - 1, -1):
        o = overlap(l, r, kern)
        s = surface_sum(o)
        if s >= min_overlap:
            break

    return round(-((-kern) ** .5)), s

def showcase(l, r, kern):
    height = l.get_height()

    ctx = create_surface_context(l.get_width() + r.get_width() - 2 * BIAS, height * 2 - BIAS)

    ctx.set_source_surface(l, 0, 0)
    ctx.paint()
    ctx.set_source_surface(r, l.get_width() - 2 * BIAS, 0)
    ctx.paint()

    ctx.set_source_surface(l, 0, height - BIAS)
    ctx.paint()
    ctx.set_source_surface(r, l.get_width() - 2 * BIAS + kern, height - BIAS)
    ctx.paint()

    return ctx.get_target()

def create_hb_font(fontfile):
    blob = hb.Blob.from_file_path(fontfile)
    face = hb.Face(blob, 0)
    font = hb.Font(face)
    font.scale = (FONT_SIZE, FONT_SIZE)
    return font

def actual_kern(l, r):
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

    return combined_advance - (l_advance + r_advance)


if __name__ == "__main__":
    import sys
    font = sys.argv[1]
    text = sys.argv[2]

    FONT_FACE = cairoft.create_cairo_font_face_for_file(font, 0)
    HB_FONT = create_hb_font(font)

    if len(text) == 1:
        surface = create_surface_for_text(text)
        surface = blur(surface, KERNEL)
        surface.write_to_png("kern.png")
        sys.exit(0)

    assert len(text) == 2

    l = create_surface_for_text('l')
    kern, sl = kern_pair(l, l, 0)
    l = create_surface_for_text('n')
    kern, sn = kern_pair(l, l, 0)
    l = create_surface_for_text('o')
    kern, so = kern_pair(l, l, 0)
    s = min(sl, sn, so)

    l = create_surface_for_text(text[0])
    r = create_surface_for_text(text[1])

    kern, s = kern_pair(l, r, s)
    font_kern = actual_kern(text[0], text[1])

    print(kern, font_kern, text)
    s = showcase(l, r, kern)
    s.write_to_png("kern.png")
