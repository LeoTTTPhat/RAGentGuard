"""mlx_lm.server with a capped Metal buffer cache (see limit_mlx_memory)."""
import os
import sys

import mlx.core as mx

mx.set_cache_limit(int(float(os.environ.get("MLX_CACHE_GB", "4")) * 1024 ** 3))
if os.environ.get("MLX_MEMORY_GB"):
    mx.set_memory_limit(int(float(os.environ["MLX_MEMORY_GB"]) * 1024 ** 3))
from mlx_lm.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
