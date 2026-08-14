#!/usr/bin/env bash
# ===========================================================================
#  One-shot build of the simulator this repository drives.
#  @tandonmitul27  --  AUTHORED FILE (new)
# ===========================================================================
#
#  WHY THIS FILE EXISTS
#      This repository deliberately does NOT vendor gem5 or DRAMsim3.  A
#      built gem5 tree is ~5 GB, and burying our changes inside a copy of
#      someone else's source makes them impossible to review.  Instead we
#      pin the upstream revisions, apply our patches as patches, and drop
#      our device configs in.  Everything this project changed about the
#      simulator is therefore visible in patches/ and configs/.
#
#  WHAT IT DOES
#      1. clone gem5 at the pinned revision            (~2 GB, few min)
#      2. clone DRAMsim3 at the pinned revision into gem5's ext/
#      3. apply patches/*.patch
#      4. install configs/dramsim3/*.ini into DRAMsim3
#      5. build DRAMsim3 (libdramsim3.so) then gem5 NULL
#
#      Re-running is safe: each step is skipped if already done.
#
#  REQUIREMENTS
#      g++ >= 10, python3 >= 3.9, scons, cmake, make, git,
#      zlib and protobuf headers.  On a machine without root, conda
#      satisfies all of them:
#          conda install -c conda-forge scons cmake zlib protobuf m4
#
#  USAGE
#      ./setup.sh              # build everything
#      ./setup.sh --jobs 8     # limit parallelism (default: all cores)
# ===========================================================================
set -euo pipefail

GEM5_URL="https://github.com/gem5/gem5.git"
GEM5_REV="cbf0eae213c5e39c727172b546434287d47b5bbe"        # v25.1.0.1
DRAMSIM_URL="https://github.com/umd-memsys/DRAMsim3.git"
DRAMSIM_REV="29817593b3389f1337235d63cac515024ab8fd6e"     # v1.0.0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEM5="$ROOT/gem5"
DRAMSIM="$GEM5/ext/dramsim3/DRAMsim3"
JOBS="$(nproc 2>/dev/null || echo 4)"
[[ "${1:-}" == "--jobs" ]] && JOBS="${2:?--jobs needs a number}"

say() { printf '\n\033[1m[setup]\033[0m %s\n' "$*"; }

# --- 1. gem5 ---------------------------------------------------------------
if [[ ! -d "$GEM5/.git" ]]; then
  say "1/5  cloning gem5 @ ${GEM5_REV:0:7} (this is the slow step)"
  git clone "$GEM5_URL" "$GEM5"
  git -C "$GEM5" checkout --quiet "$GEM5_REV"
else
  say "1/5  gem5 already present, skipping clone"
fi

# --- 2. DRAMsim3 -----------------------------------------------------------
if [[ ! -d "$DRAMSIM/.git" ]]; then
  say "2/5  cloning DRAMsim3 @ ${DRAMSIM_REV:0:7}"
  mkdir -p "$(dirname "$DRAMSIM")"
  git clone "$DRAMSIM_URL" "$DRAMSIM"
  git -C "$DRAMSIM" checkout --quiet "$DRAMSIM_REV"
else
  say "2/5  DRAMsim3 already present, skipping clone"
fi

# --- 3. our patches --------------------------------------------------------
say "3/5  applying patches/"
for p in "$ROOT"/patches/*.patch; do
  [[ -e "$p" ]] || continue
  if git -C "$GEM5" apply --check "$p" 2>/dev/null; then
    git -C "$GEM5" apply "$p"
    echo "       applied  $(basename "$p")"
  else
    echo "       already applied (or N/A)  $(basename "$p")"
  fi
done

# --- 4. our device configs -------------------------------------------------
say "4/5  installing device configs into DRAMsim3"
cp -v "$ROOT"/configs/dramsim3/*.ini "$DRAMSIM/configs/" | sed 's/^/       /'

# --- 5. build --------------------------------------------------------------
say "5/5  building DRAMsim3, then gem5 NULL  (-j$JOBS)"
if [[ ! -f "$DRAMSIM/libdramsim3.so" ]]; then
  ( cd "$DRAMSIM" && mkdir -p build && cd build \
      && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j"$JOBS" )
  # gem5 expects the library beside the sources
  [[ -f "$DRAMSIM/build/libdramsim3.so" ]] && \
      cp "$DRAMSIM/build/libdramsim3.so" "$DRAMSIM/"
else
  echo "       libdramsim3.so present, skipping"
fi
( cd "$GEM5" && scons build/NULL/gem5.opt -j"$JOBS" )

cat <<EOF

===========================================================================
 Build complete.

   gem5 binary : $GEM5/build/NULL/gem5.opt
   DRAMsim3    : $DRAMSIM

 Next:  make check-fast     # ~2 min, verifies the whole tree
        make check          # ~5 min, adds the full-stack runs
===========================================================================
EOF
