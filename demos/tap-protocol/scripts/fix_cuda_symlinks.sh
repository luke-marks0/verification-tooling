#!/usr/bin/env bash
# Vast.ai bind-mounts driver libs at /usr/lib/x86_64-linux-gnu/libcuda.so.<X.Y.Z>
# but does not create the libcuda.so.1 / libcuda.so symlinks torch dlopens.
# Verbatim from /home/jon/.claude/CLAUDE.md vast section.
set -eu

cd /usr/lib/x86_64-linux-gnu
for f in libcuda libcudadebugger libnvidia-ml libnvidia-nvvm libnvidia-ptxjitcompiler \
         libnvcuvid libnvidia-opencl libnvidia-cfg libnvidia-gpucomp \
         libnvidia-opticalflow libnvidia-sandboxutils; do
    # strict glob: X.Y.Z only, avoids self-loop on already-existing libcuda.so.1
    src=$(ls ${f}.so.[0-9]*.[0-9]*.[0-9]* 2>/dev/null | head -1)
    rm -f "${f}.so.1" "${f}.so"
    if [ -n "$src" ]; then
        ln -sf "$src" "${f}.so.1"
        ln -sf "$src" "${f}.so"
    fi
done

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
echo "cuda symlinks fixed; LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
