# halfkern

HalfKern is essentially a font auto-kerning tool masqueraded as a
kerning audit tool.

The way the tool works is that for every pair of letters that
are considered, it will blur their renderings and space the
two such that the blurred images overlap a certain amount.
This certain amount is found by first calibrating using the
"ll", "nn", and "oo" pairs.

The tool currently does _not_ store autokerning results in the font.

## Usage

```
$ python3 kern_pair.py FontFile.ttf --dict dictionary.txt
$ python3 kern_pair.py FontFile.ttf PairString
```

## Example

```
$ python3 kern_pair.py Roboto-Regular.ttf --dict /usr/share/dict/words -l
fi 0 -4
yt 4 0
Te -9 -5
To -9 -5
TA -8 -4
DT -5 -1
```

The first value is the pair of letter to kern. The second value is, in EM units, the suggested kerning value, and the last value is the kerning currently
in the font.  Only pairs where the two kerning values differ by a tolerance
amount are showed.  This tolerance can be set using `-t` or `--tolerance`.
The default tolerance is 3.3%.

The `-l` or `--letters-only` makes the tool only consider kerning between
two letters (ie. no punctuation).  The tool also ignores digits, since they
typically have a fixed width and no kerning by design.

A file `kerned.pdf` is always generated, with each page showcasing one pair
the tool thinks would need adjustment: [kerned.pdf](/images/kerned.pdf)

To inspect the pairs reported, you can use the `kern_pair.py` tool again:
```
$ python3 kern_pair.py Roboto-Regular.ttf To
To autokern: -9 (-184 units) existing kern: -5 (-99 units)
Saving kern.png
```
In this case the tool thinks the pair "To" is not kerned enough in Roboto.
Obviously that's up to taste. But here's the two files `kern.png`
and `kerned.pdf` generated by the tool:

![kern.png](/images/kern.png)

In the `kern.png` image, the first line is with no kerning. The second line
is the tool's suggestion, and the third line is the existing font
kerning.  The `kerned.pdf`, the pair is showcased between lower, and upper,
letters.  The three rows, similarly, show no-kerning, tool's suggestion,
and existing kern.


## Algorithms

The tool has two different ways to form an envelope around each glyph.
This can be set using `--envelope sdf` (default) or `--envelope gaussian`.

It also has two different ways to summarize the overlap of two glyph envelopes.
This can be set using `--reduce sum` (default) or `--reduce max`.

This gives four different combinations of modes to run the tool.  Which
one works best for a project is subjective and should be experimented with.
The defaults, in my opinion, generate the best results


## Dictionary

To produce per-language dictionaries to be used with this tool you can use the
[aosp-test-texts](https://github.com/googlefonts/aosp-test-texts)
repository, or the
[libreoffice spellcheck dictionaries](https://cgit.freedesktop.org/libreoffice/dictionaries/),
or the
[harfbuzz-wikipedia-testing](https://github.com/harfbuzz/harfbuzz-testing-wikipedia).

*TODO:* Expand on how to use these.

For simple English wordlist on Linux and Mac platforms you can use
`/usr/share/words/dict`.


## Debugging

To see the envelope for one character, use:
```
$ python3 kern_pair.py fontfile.ttf X
```
This will generate the envelope image for `X` and save it to `envelope.png`.
