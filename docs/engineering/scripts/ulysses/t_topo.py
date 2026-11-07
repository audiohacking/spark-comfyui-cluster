import logging, sys
logging.disable(logging.WARNING)
fams = ["minimax_h3", "krea2", "chroma", "flux", "zimage", "omnigen2", "wan", "unknown"]
from dgx_monarch.topology import choose_auto_topology
for fam in fams:
    for w in range(2, 9):
        for cfg in (1.0, 4.0):
            for b in (1, 2):
                try:
                    d = choose_auto_topology(fam, "int8", 1.0, w, cfg_value=cfg, batch_size=b)
                    t = d.topology
                    r = f"u{t.ulysses} r{t.ring} c{t.cfg} dp{t.dp} f{int(t.fsdp)}"
                except Exception as e:
                    r = "ERR " + type(e).__name__
                print(fam, w, cfg, b, r)
