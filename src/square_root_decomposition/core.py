import math
from typing import Callable, List, Sequence


class Operation:
    """Aggregate operation bundles used by :class:`SqrtDecomposition`.

    A decomposition needs three pieces per operation: how to merge two
    values, what counts as an identity (so a partial block merge has a
    starting point), and whether the aggregate is defined for an empty
    range. Sum, product, min, and max cover the common cases; each is
    implemented as a small frozen dataclass-like object so callers can pass
    a single name around rather than three separate callables.
    """

    def __init__(
        self,
        name: str,
        merge: Callable[[float, float], float],
        identity: float,
        supports_empty: bool,
    ) -> None:
        self.name = name
        self.merge = merge
        self.identity = identity
        self.supports_empty = supports_empty

    def __repr__(self) -> str:  # makes failure messages readable in tests
        return f"Operation({self.name!r})"


# Pre-built operations. They intentionally use Python floats throughout so the
# library can accept int or float elements without per-call type juggling.
SUM = Operation("sum", lambda a, b: a + b, 0.0, supports_empty=True)
PRODUCT = Operation("product", lambda a, b: a * b, 1.0, supports_empty=True)
MIN = Operation("min", lambda a, b: a if a <= b else b, math.inf, supports_empty=False)
MAX = Operation("max", lambda a, b: a if a >= b else b, -math.inf, supports_empty=False)


class SqrtDecomposition:
    """Block decomposition of a list supporting point updates and range queries.

    The list is split into ``ceil(n / block_size)`` contiguous blocks of size
    roughly ``block_size``. Each block stores a precomputed aggregate of its
    elements. A range query touches at most two partial blocks element by
    element plus at most ``ceil(n / block_size)`` full blocks by their stored
    aggregate, giving ``O(sqrt n)`` per query and ``O(sqrt n)`` worst case
    per update (when the block aggregate must be recomputed). Construction
    is ``O(n)``.

    The default block size is ``ceil(sqrt(n))``. Pass an explicit
    ``block_size`` to tune the constant factor for workloads whose
    query/update ratio differs; the asymptotic bound only holds when
    ``block_size`` is ``Theta(sqrt n)``.

    Edge cases the implementation commits to:
      * Empty container (length 0) has no blocks; the first update grows it.
      * Trailing block may be shorter than ``block_size``.
      * Single-element and single-block arrays work for any operation whose
        identity is a valid element (sum/product) — min/max raise on empty.
      * Identity-based operations (sum, product) answer empty ranges with
        their identity; min/max raise ``ValueError`` on an empty range.
      * Updates with index exactly ``len(self)`` are allowed and append; any
        larger index raises ``IndexError``.
    """

    def __init__(
        self,
        data: Sequence[float],
        operation: Operation,
        block_size: int | None = None,
    ) -> None:
        self._data: List[float] = list(data)
        self._op = operation
        n = len(self._data)

        # Special-casing n == 0 keeps the rest of the code free of division
        # by zero; the lazy-growth branch in `update` knows how to rebuild
        # from an empty start.
        if n == 0:
            self._block_size = 1
            self._num_blocks = 0
            self._blocks: List[float] = []
            return

        if block_size is None:
            block_size = max(1, math.ceil(math.sqrt(n)))
        if block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self._block_size = block_size
        self._num_blocks = math.ceil(n / block_size)
        self._blocks = [self._op.identity] * self._num_blocks
        self._rebuild_all()

    def _rebuild_all(self) -> None:
        op = self._op
        data = self._data
        bs = self._block_size
        for b in range(self._num_blocks):
            start = b * bs
            end = min(start + bs, len(data))
            agg = op.identity
            for i in range(start, end):
                agg = op.merge(agg, data[i])
            self._blocks[b] = agg

    def _block_index(self, i: int) -> int:
        return i // self._block_size

    def _rebuild_block(self, b: int) -> None:
        op = self._op
        data = self._data
        bs = self._block_size
        start = b * bs
        end = min(start + bs, len(data))
        agg = op.identity
        for i in range(start, end):
            agg = op.merge(agg, data[i])
        self._blocks[b] = agg

    def __len__(self) -> int:
        return len(self._data)

    def query(self, left: int, right: int) -> float:
        """Aggregate ``self.data()[left:right]`` (half-open, Python slicing).

        Supports empty ranges iff the operation has an identity; for min/max
        an empty range raises ``ValueError``. Out-of-range indices are
        silently clamped to the valid range, matching list slicing semantics.
        """
        n = len(self._data)
        left = max(0, min(left, n))
        right = max(0, min(right, n))
        if right < left:
            right = left
        if left == right:
            if not self._op.supports_empty:
                raise ValueError(
                    f"operation {self._op.name!r} is undefined for an empty range"
                )
            return self._op.identity

        op = self._op
        bs = self._block_size
        result = None
        i = left
        # Accumulate the leading partial block element by element.
        while i < right and i % bs != 0:
            result = self._data[i] if result is None else op.merge(result, self._data[i])
            i += 1
        # Then take whole blocks via their cached aggregate.
        while i + bs <= right:
            result = self._blocks[i // bs] if result is None else op.merge(result, self._blocks[i // bs])
            i += bs
        # Finally the trailing partial block, again element by element.
        while i < right:
            result = self._data[i] if result is None else op.merge(result, self._data[i])
            i += 1
        return result

    def update(self, index: int, value: float) -> None:
        """Set ``self.data()[index] = value``.

        ``index == len(self)`` appends a new element. ``block_size`` is
        never resized on append: appends past a full block boundary simply
        allocate a new trailing block.
        """
        n = len(self._data)
        if index < 0 or index > n:
            raise IndexError(
                f"update index {index} is out of range for length {n}"
            )
        if index == n:
            self._data.append(value)
            b = self._block_index(index)
            if b >= self._num_blocks:
                # New block needed; allocate it at identity and recompute so
                # the trailing-block invariant (block == aggregate of its
                # elements) is preserved even when this is the very first
                # element of an originally empty container.
                self._num_blocks = b + 1
                if len(self._blocks) < self._num_blocks:
                    self._blocks.append(self._op.identity)
                self._rebuild_block(b)
            else:
                self._rebuild_block(b)
            return

        self._data[index] = value
        self._rebuild_block(self._block_index(index))

    def data(self) -> List[float]:
        """Return a copy of the underlying element list."""
        return list(self._data)

    def __repr__(self) -> str:
        return (
            f"SqrtDecomposition(len={len(self._data)}, "
            f"op={self._op.name!r}, block_size={self._block_size})"
        )
