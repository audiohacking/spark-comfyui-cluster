"""Compare the model folders of the cluster's Sparks by name.

Called by spark-comfyui.sh as:
    cluster_models.py list HOSTS [--missing] [PATTERN ...]
    cluster_models.py plan HOSTS SOURCE PATTERN [PATTERN ...]

HOSTS is a comma-separated list of host labels, in display order. stdin
carries every host's index, one file per line:
    <host>\t<path relative to models/>\t<size in bytes>

A PATTERN with a glob character (* ? [) is matched against the relative path
and against the file name; any other PATTERN is a substring of the relative
path. Matching ignores case. No PATTERN in 'list' means every file.

Emits:
    list  ->  a table: one row per file, one size column per host
    plan  ->  COPY\t<dest>\t<path>\t<size>                  missing on dest
              DIFF\t<dest>\t<path>\t<src size>\t<dest size>  same name, other size
              (only files that SOURCE has; exit 1 when it has none matching)
"""
import fnmatch, sys


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return ("%d%s" % (n, unit)) if unit == "B" else ("%.1f%s" % (n, unit))
        n /= 1024.0


def matcher(patterns):
    pats = [p.lower() for p in patterns]

    def match(path):
        if not pats:
            return True
        low = path.lower()
        base = low.rsplit("/", 1)[-1]
        for p in pats:
            if any(ch in p for ch in "*?["):
                if fnmatch.fnmatchcase(low, p) or fnmatch.fnmatchcase(base, p):
                    return True
            elif p in low:
                return True
        return False
    return match


def read_index(hosts, match):
    # files[path][host] = size
    files = {}
    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3 or parts[0] not in hosts or not parts[2].isdigit():
            continue
        host, path, size = parts
        if match(path):
            files.setdefault(path, {})[host] = int(size)
    return files


def cmd_list(hosts, args):
    missing_only = "--missing" in args
    patterns = [a for a in args if a != "--missing"]
    files = read_index(hosts, matcher(patterns))
    rows, odd = [], 0
    for path in sorted(files):
        have = files[path]
        sizes = set(have.values())
        complete = len(have) == len(hosts) and len(sizes) == 1
        if not complete:
            odd += 1
        if missing_only and complete:
            continue
        cells = []
        for h in hosts:
            if h not in have:
                cells.append("-")
            elif len(sizes) > 1:
                cells.append(human(have[h]) + " !=")
            else:
                cells.append(human(have[h]))
        rows.append([path] + cells)
    head = ["model"] + hosts
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    fmt = "  " + "  ".join("%%-%ds" % w if i == 0 else "%%%ds" % w
                           for i, w in enumerate(widths))
    print(fmt % tuple(head))
    for r in rows:
        print(fmt % tuple(r))
    print()
    print("  %d file(s) matched: %d identical on all %d host(s), %d missing "
          "somewhere or differing (- missing, != other size)"
          % (len(files), len(files) - odd, len(hosts), odd))


def cmd_plan(hosts, args):
    if len(args) < 2:
        sys.exit("cluster_models.py plan HOSTS SOURCE PATTERN [PATTERN ...]")
    src, patterns = args[0], args[1:]
    files = read_index(hosts, matcher(patterns))
    found = False
    for path in sorted(files):
        have = files[path]
        if src not in have:
            continue
        found = True
        for h in hosts:
            if h == src:
                continue
            if h not in have:
                print("COPY\t%s\t%s\t%d" % (h, path, have[src]))
            elif have[h] != have[src]:
                print("DIFF\t%s\t%s\t%d\t%d" % (h, path, have[src], have[h]))
    sys.exit(0 if found else 1)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("list", "plan"):
        sys.exit("usage: cluster_models.py list|plan HOSTS [ARGS ...]")
    hosts = [h for h in sys.argv[2].split(",") if h]
    (cmd_list if sys.argv[1] == "list" else cmd_plan)(hosts, sys.argv[3:])


main()
