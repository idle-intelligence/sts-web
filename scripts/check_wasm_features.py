#!/usr/bin/env python3
"""Fail if a built wasm is missing a required target feature.

Reads the `target_features` custom section that rustc/LLVM writes into each
wasm module (a list of `+feature` entries) and checks it against the
features given on the command line.

    check_wasm_features.py FILE.wasm +simd128 [+atomics ...] [-- FILE2.wasm +simd128 ...]
"""
import sys


def leb128(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return result, pos


def target_features(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != b"\0asm":
        raise SystemExit(f"error: {path} is not a wasm module")
    pos = 8
    while pos < len(buf):
        section_id = buf[pos]
        size, pos = leb128(buf, pos + 1)
        end = pos + size
        if section_id == 0:
            name_len, p = leb128(buf, pos)
            name = buf[p:p + name_len].decode()
            p += name_len
            if name == "target_features":
                count, p = leb128(buf, p)
                feats = []
                for _ in range(count):
                    prefix = chr(buf[p])
                    n, p = leb128(buf, p + 1)
                    feats.append(prefix + buf[p:p + n].decode())
                    p += n
                return feats
        pos = end
    return []


def main(argv):
    groups, cur = [], []
    for a in argv:
        if a == "--":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    groups.append(cur)

    failed = False
    for g in groups:
        if not g:
            continue
        path, required = g[0], g[1:]
        feats = target_features(path)
        missing = [r for r in required if r not in feats]
        status = "MISSING " + " ".join(missing) if missing else "ok"
        print(f"{path}: {' '.join(sorted(feats)) or '(no target_features section)'} -> {status}")
        failed |= bool(missing)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
