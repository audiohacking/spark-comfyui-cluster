"""Read spark-cluster.toml (dgx-monarch's cluster.toml) for the shell.

Called by spark-comfyui.sh as:
    cluster_read.py PATH hosts                  one row per [[hosts]] entry
    cluster_read.py PATH local IP [IP ...]      the entry whose address is local
    cluster_read.py PATH check SHIM             container-mode sanity warnings

Emits tab-separated rows:
    hosts  ->  <name>\t<address>\t<ssh_user or empty>
    local  ->  <name>\t<address>          (exit 1 when no entry is local)
    check  ->  WARN\t<message>            (one per problem; exit 0 either way)

Host-side only, on the host's own python3 (stdlib tomllib, Python 3.11+;
DGX OS ships 3.12). dgx-monarch validates the full schema itself when the
driver loads the file; this reads just the fields the container tooling
needs and refuses a file it cannot parse. Exits non-zero with
"cluster config error: ..." on stderr.
"""
import ipaddress, re, sys

try:
    import tomllib
except ImportError:  # Python < 3.11
    print("cluster config error: the host python3 is older than 3.11 (no tomllib)",
          file=sys.stderr)
    sys.exit(1)

ADDR = re.compile(r"^tcp://(\d{1,3}(?:\.\d{1,3}){3}):(\d{1,5})$")
DEFAULT_PORT = 26600


def die(msg):
    print("cluster config error: %s" % msg, file=sys.stderr)
    sys.exit(1)


def clean(value, what):
    # Every value lands in a TSV row and then in shell variables, so a tab or
    # newline would split a row; refuse rather than mangle.
    value = str(value)
    if any(ch in value for ch in "\t\r\n"):
        die("%s contains a tab or newline" % what)
    return value


def hosts(cfg):
    rows = []
    for i, h in enumerate(cfg.get("hosts") or []):
        if not isinstance(h, dict) or not h.get("name"):
            die("[[hosts]] entry %d has no name" % (i + 1))
        name = clean(h["name"], "host name")
        addr = h.get("address")
        if not addr:
            # dgx-monarch's own default: a literal-IP name binds on :26600.
            try:
                ipaddress.IPv4Address(name)
            except ValueError:
                die("host '%s' needs an address (its name is not an IPv4 literal)" % name)
            addr = "tcp://%s:%d" % (name, DEFAULT_PORT)
        addr = clean(addr, "host address")
        if not ADDR.match(addr):
            die("host '%s' address must be tcp://<fabric-ipv4>:<port>, got '%s'" % (name, addr))
        rows.append((name, addr, clean(h.get("ssh_user", ""), "ssh_user")))
    if not rows:
        die("no [[hosts]] entries")
    return rows


def main():
    if len(sys.argv) < 3:
        die("usage: cluster_read.py PATH hosts|local|check [ARGS]")
    path, cmd, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    try:
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        die("%s: %s" % (path, e))

    if cmd == "hosts":
        for row in hosts(cfg):
            print("\t".join(row))
    elif cmd == "local":
        local = set(args)
        for name, addr, _ in hosts(cfg):
            if ADDR.match(addr).group(1) in local:
                print("%s\t%s" % (name, addr))
                return
        sys.exit(1)
    elif cmd == "check":
        hosts(cfg)
        cluster = cfg.get("cluster") or {}
        shim = args[0] if args else ""
        if shim and cluster.get("python") != shim:
            print("WARN\tcluster.python should be \"%s\" in container mode "
                  "(the per-host probes run that path)" % shim)
        # dgx-monarch defaults auto_heal to TRUE, so an absent key is a warning too.
        if cluster.get("auto_heal", True):
            print("WARN\tcluster.auto_heal should be false in container mode: dgx-monarch "
                  "would try to restart a bare-metal worker; the container restart policy heals instead")
        if cluster.get("transport_security") != "trusted_fabric":
            print("WARN\tcluster.transport_security must be \"trusted_fabric\" or every attach is refused")
        if not cluster.get("client_bind"):
            print("WARN\tcluster.client_bind is missing: the attach refuses without a fabric bind")
    else:
        die("unknown command '%s'" % cmd)


main()
