#!/usr/bin/env python3
"""
Dynamic Bit-Window Pollard-Rho with ``tame`` (random or range) and ``wild`` workers.

This script integrates Pollard's rho, Pollard's kangaroo, and a Chinese Remainder
Theorem (CRT) post-processing step into the BitCrack repository.  It can be used as a
standalone tool to accelerate private key searches for addresses by combining
partial information from multiple workers.

The script supports two backends:
  * ``ecdsa`` – pure Python implementation using ``python-ecdsa`` (default)
  * ``ice``   – bindings to the secp256k1 C implementation used by BitCrack

Bit offsets are counted from the least-significant bit (LSB=0), matching the
ordering used by the C++ Pollard implementation.

Example usage::

    python3 tools/pollard_kangaroo_crt.py \
        --backend ice \
        --offsets_count 7 --window_size 40 --max_steps 100000 \
        --workers 96 \
        --target_rmd af0e8dcf38b36ceb4b5806efc6cbb04586903c17 \
        --L 1 --U 1000000 \
        --tames 45

"""
import argparse, hashlib, sys, math, random
from multiprocessing import Pool, current_process

# --- pure-Python fallback via python-ecdsa ---
from ecdsa import SigningKey, SECP256k1
ECDSA_ORDER = SECP256k1.order

def py_scalar_multiplication(k: int) -> bytes:
    sk = SigningKey.from_secret_exponent(k, curve=SECP256k1)
    vk = sk.verifying_key
    # compressed = prefix + x
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    prefix = b'\x02' if (y % 2) == 0 else b'\x03'
    return prefix + int(x).to_bytes(32, 'big')

def py_privatekey_to_h160(k: int) -> str:
    pub = py_scalar_multiplication(k)
    sha = hashlib.sha256(pub).digest()
    return hashlib.new('ripemd160', sha).hexdigest()


# placeholders — we'll bind these after reading --backend
scalar_multiplication = None    # (k:int) -> bytes
privatekey_to_h160    = None    # (k:int) -> hex str
N = ECDSA_ORDER
backend = None


def compress_pubkey(upub: bytes) -> bytes:
    """
    If ``upub`` is a 65-byte uncompressed key (0x04||X||Y), turn it into 33-byte
    compressed form (0x02/0x03||X).  Otherwise, return ``upub`` unchanged.
    """
    if len(upub) == 65 and upub[0] == 0x04:
        x = upub[1:33]
        y = upub[33:65]
        prefix = b'\x02' if (y[-1] % 2) == 0 else b'\x03'
        return prefix + x
    return upub


def load_ice_backend():
    """Import secp256k1 C binding and wrap it to match our API."""
    try:
        import secp256k1 as ice
    except ImportError:
        print("[WARN] `secp256k1` module not found; falling back to pure Python backend")
        return None, None

    # pick up curve order if provided
    global N
    N = getattr(ice, 'N', None) or getattr(ice, 'ORDER', None) or N

    def ice_scalar(k: int) -> bytes:
        # ice.scalar_multiplication may return 65-byte uncompressed
        upub = ice.scalar_multiplication(k)
        return compress_pubkey(upub)

    def ice_h160(k: int) -> str:
        pub = ice_scalar(k)
        sha = hashlib.sha256(pub).digest()
        return hashlib.new('ripemd160', sha).hexdigest()

    return ice_scalar, ice_h160


# -------------------------------------------------------------------
# CRT + candidate enumeration + worker pool are unchanged:
# -------------------------------------------------------------------

def crt(constraints):
    x, M = 0, 1
    for mod, rem in constraints:
        g = math.gcd(M, mod)
        if (rem - x) % g:
            print(f"[WARN] Skipping constraint mod={mod}, rem={rem}")
            continue
        M_g, mod_g = M // g, mod // g
        inv = pow(M_g, -1, mod_g)
        t = ((rem - x) // g * inv) % mod_g
        x += M * t
        M *= mod_g
        x %= M
    return x, M

def enumerate_candidates(k0, M, L, U):
    tmin = math.ceil((L - k0) / M)
    tmax = math.floor((U - k0) / M)
    return [k0 + t*M for t in range(tmin, tmax+1) if L <= k0 + t*M <= U]


def worker(args):
    target, ws, offsets, mask, L_i, U_i, max_steps, direction, verbose = args
    name = current_process().name

    # precompute the bits we want to match (offsets are measured from the
    # least-significant bit, consistent with the C++ Pollard engine):
    want = {off: (target >> off) & mask for off in offsets}
    found = {}
    span = U_i - L_i + 1
    steps = min(max_steps, span)

    if verbose:
        print(f"[{name}] {direction} [{L_i},{U_i}] steps={steps}")

    for i in range(steps):
        a = (L_i + i) if direction=='tame' else (U_i - i)
        pub = scalar_multiplication(a)
        sha = hashlib.sha256(pub).digest()
        rmd = hashlib.new('ripemd160', sha).digest()
        val = int.from_bytes(rmd, 'little')

        for off, w in want.items():
            if off in found:
                continue
            if ((val >> off) & mask) == w:
                found[off] = (a >> off) & mask
                if verbose:
                    print(f"[{name}] matched off={off} -> {found[off]}")

        if len(found) == len(offsets):
            break

    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['ecdsa','ice'], default='ecdsa',
                   help="Which ECC backend to use (default=ecdsa)")
    p.add_argument('--target_rmd',   required=True)
    p.add_argument('--L',            type=int, required=True)
    p.add_argument('--U',            type=int, required=True)
    p.add_argument('--offsets_count',type=int, default=4)
    p.add_argument('--offsets',      type=str,
                   help="Comma-separated bit offsets (LSB=0)")
    p.add_argument('--window_size',  type=int)
    p.add_argument('--max_steps',    type=int)
    p.add_argument('--workers',      type=int, default=4)
    p.add_argument('--tames',        type=int)
    p.add_argument('--verbose','-v', action='store_true')
    p.add_argument('--full', action='store_true',
                   help="Relaunch walks until a key is found or interrupted")
    args = p.parse_args()

    # bind the chosen backend
    global scalar_multiplication, privatekey_to_h160, backend
    if args.backend == 'ice':
        scalar_multiplication, privatekey_to_h160 = load_ice_backend()
        if scalar_multiplication is None:
            scalar_multiplication = py_scalar_multiplication
            privatekey_to_h160   = py_privatekey_to_h160
            backend = 'ecdsa'
        else:
            backend = 'ice'
    else:
        scalar_multiplication = py_scalar_multiplication
        privatekey_to_h160   = py_privatekey_to_h160
        backend = 'ecdsa'

    L, U = args.L, args.U
    if not (0 <= L <= U < N):
        print(f"[ERROR] invalid range [{L},{U}] for curve order N={N}")
        sys.exit(1)

    span = U - L + 1
    bits = span.bit_length()
    ws = args.window_size or max(1, min(
        math.ceil(bits/args.offsets_count),
        160//args.offsets_count
    ))
    mask = (1 << ws) - 1

    if args.offsets:
        offsets = list(map(int, args.offsets.split(',')))
    else:
        max_off = 160 - ws
        step    = max(1, max_off // (args.offsets_count - 1))
        offsets = [i*step for i in range(args.offsets_count)]
    # Offsets are measured from the least-significant bit, consistent with the C++ Pollard engine.

    max_steps = args.max_steps or span
    # ensure we never do more than the subrange length
    max_steps = min(max_steps, span)

    n = args.workers
    n_tame = min(args.tames, n) if args.tames else n//2
    n_wild = n - n_tame

    if args.verbose:
        print(f"backend={backend}, N={N}")
        print(f"Range=[{L},{U}], ws={ws}, offsets={offsets}")
        print(f"workers={n} (tame={n_tame}, wild={n_wild}), max_steps={max_steps}")

    target = int.from_bytes(bytes.fromhex(args.target_rmd), 'little')
    chunk  = span // n
    constraints = {}

    while True:
        jobs = []

        # tame jobs
        if args.full or args.tames:
            for _ in range(n_tame):
                st = random.randint(L, U)
                en = min(st + max_steps - 1, U)
                jobs.append((target, ws, offsets, mask, st, en,
                             max_steps, 'tame', args.verbose))
        else:
            for i in range(n_tame):
                st = L + i*chunk
                en = min(L + (i+1)*chunk - 1, U)
                jobs.append((target, ws, offsets, mask, st, en,
                             max_steps, 'tame', args.verbose))

        # wild jobs
        if args.full:
            for _ in range(n_wild):
                en = random.randint(L, U)
                st = max(L, en - max_steps + 1)
                jobs.append((target, ws, offsets, mask, st, en,
                             max_steps, 'wild', args.verbose))
        else:
            for j in range(n_wild):
                en = U - j*chunk
                st = max(L, en - chunk + 1)
                jobs.append((target, ws, offsets, mask, st, en,
                             max_steps, 'wild', args.verbose))

        # collect CRT constraints
        with Pool(n) as pool:
            for res in pool.imap_unordered(worker, jobs):
                for off, lo in res.items():
                    if args.full or off not in constraints:
                        constraints[off] = lo
                        if args.verbose:
                            print(f"[MAIN] collected off={off} -> {lo}")
                if len(constraints) == len(offsets):
                    # Wait for workers to exit to avoid orphaned GPU tasks
                    pool.close()
                    pool.join()
                    break

        if len(constraints) < len(offsets):
            if args.full:
                continue
            print(f"[ERROR] only {len(constraints)}/{len(offsets)} offsets found")
            sys.exit(1)

        conlist = [
            (1 << (off + ws),
             (lo << off) & ((1 << (off + ws)) - 1))
            for off, lo in constraints.items()
        ]
        k0, Mval = crt(conlist)
        print(f"CRT k≡{k0} mod {Mval}")

        cands = enumerate_candidates(k0, Mval, L, U)
        print(f"candidates: {len(cands)}")

        found = False
        for k in cands:
            h = privatekey_to_h160(k)
            print(f"test {k} -> {h}")
            if h.lower() == args.target_rmd.lower():
                print("FOUND key:", k)
                with open('KEYFOUND.TXT','a') as f:
                    f.write(f"{k}\n")
                found = True
                break

        if found:
            sys.exit(0)

        if args.full:
            print("no key found, relaunching walks...")
            continue

        print("no key found")
        break

if __name__ == '__main__':
    main()
