#
# Square root decomposition container.
#
# Only the core class is re-exported here so that:
#     from square_root_decomposition import SqrtDecomposition
# works and exposes nothing else. Keeping the surface tiny makes upgrades
# safe and the README honest about the real API.
#
from .core import SqrtDecomposition, Operation

__all__ = ["SqrtDecomposition", "Operation"]
