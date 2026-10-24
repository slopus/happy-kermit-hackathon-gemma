"""Word error rate with light normalization (lowercase, mister->mr, 10->ten)."""
import re


def norm(s):
    s = s.lower().replace("mister", "mr").replace("10", "ten")
    return re.sub(r"[^a-z' ]", " ", s).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)
