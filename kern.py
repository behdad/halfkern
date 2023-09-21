import cairo as cr
import functools

KERNEL_WIDTH = 16
KERNEL = list(range(KERNEL_WIDTH))
KERNEL += list(KERNEL[-2::-1])
BIAS = len(KERNEL) // 2

FONT_FAMILY = "Roboto Regular"
FONT_SIZE = 64

def blur(surface, kernel):
    s = sum(kernel)
    kernel = [x/s for x in kernel]

    data = surface.get_data()
    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    s1 = cr.ImageSurface.create_for_data(data, cr.FORMAT_A8, width, height, stride)
    d1 = s1.get_data()
    for i in range(height):
        for j in range(width):
            p = 0
            for k,v in enumerate(kernel):
                if j + k - BIAS >= 0 and j + k - BIAS < width:
                    p += data[i*stride + j + k - BIAS] * v
            d1[i*stride + j] = int(p)

    s2 = cr.ImageSurface.create_for_data(data, cr.FORMAT_A8, width, height, stride)
    d2 = s2.get_data()
    for j in range(width):
        for i in range(height):
            p = 0
            for k,v in enumerate(kernel):
                if i + k - BIAS >= 0 and i + k - BIAS < height:
                    p += d1[(i + k - BIAS) * stride + j] * v
            d2[i*stride + j] = int(p)

    return s2

def create_surface_context(width, height):
    surface = cr.ImageSurface(cr.FORMAT_A8, width, height)
    ctx = cr.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.select_font_face(FONT_FAMILY, cr.FONT_SLANT_NORMAL, cr.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(FONT_SIZE)
    return ctx

@functools.cache
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

    width = 2 * BIAS + kern

    ctx = create_surface_context(width, height)

    ctx.set_source_surface(r, 0, 0)
    ctx.set_operator(cr.OPERATOR_SOURCE)
    ctx.paint()

    ctx.set_source_surface(l, -l.get_width() + BIAS + kern, 0)
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


if __name__ == "__main__":
    import sys
    text = sys.argv[1]

    if len(text) == 1:
        surface = create_surface_for_text(text)
        surface = blur(surface, KERNEL)
        surface.write_to_png("kern.png")
        sys.exit(0)

    assert len(text) == 2

    l = blur(create_surface_for_text(text[0]), KERNEL)
    r = blur(create_surface_for_text(text[1]), KERNEL)

    for kern in range(0, 2 * BIAS + 1):
        o = overlap(l, r, kern)
        s = surface_sum(o)
        if s:
            print(kern)
            break
