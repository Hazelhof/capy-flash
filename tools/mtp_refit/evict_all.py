# posix_fadvise DONTNEED on the clean page cache of models, dumps and downloads (files >= 1 MiB under the given
# directories): otherwise the ~65 GB pinned arena of Flash-Next can stall in reclaim and systemd-oomd kills the server.
import os
import sys

roots = sys.argv[1:]  # directories holding models, dumps and downloads
n = tot = 0
for r in roots:
    for d, _, fs in os.walk(r):
        for f in fs:
            p = os.path.join(d, f)
            try:
                if os.path.islink(p) or os.path.getsize(p) < (1 << 20):
                    continue
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
                tot += os.path.getsize(p)
            except OSError:
                pass
print(f"evict: {n} files ({tot / 2**30:.0f} GiB)")
