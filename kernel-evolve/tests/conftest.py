"""Shared pytest configuration for the kernel-evolve suite."""

import importlib.util

# The GMM FP8 standalone script needs the TPU-side qwix and tokamax packages; skip
# collecting it on machines (CPU, CI) that do not have them installed.
collect_ignore = []
if importlib.util.find_spec("qwix") is None or importlib.util.find_spec("tokamax") is None:
  collect_ignore.append("standalone_gmm_fp8_blockwise_test.py")
