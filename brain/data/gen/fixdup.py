"""fixdup.py FILE 'LINE|old|new' ... : replace the heard text on a given source line (dedupe helper)."""
import sys
p = sys.argv[1]; L = open(p).read().split("\n")
for spec in sys.argv[2:]:
    n, old, new = spec.split("|")
    n = int(n)
    assert f"## {old} ##" in L[n - 1], (n, old)
    L[n - 1] = L[n - 1].replace(f"## {old} ##", f"## {new} ##", 1)
open(p, "w").write("\n".join(L))
