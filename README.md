# Square Root Decomposition

A small Python library that decomposes a list into fixed-size blocks so point updates and range aggregate queries both run in O(sqrt n).

```python
from square_root_decomposition import SqrtDecomposition
from square_root_decomposition.core import SUM, PRODUCT, MIN, MAX, Operation

s = SqrtDecomposition([1, 2, 3, 4, 5, 6, 7, 8], SUM)
s.query(0, 8)        # 36
s.update(3, 100)
s.query(0, 8)        # 132
s.query(2, 5)         # 104

# Custom aggregate, e.g. bitwise XOR:
XOR = Operation("xor", lambda a, b: a ^ b, 0, supports_empty=True)
sx = SqrtDecomposition([1, 2, 3, 4, 5], XOR)
sx.query(0, 5)        # 1
```

## Why

Square root decomposition is the middle ground between a flat list (O(1) update, O(n) query) and a segment tree (O(log n) both, but heavier code and a fixed operation per tree). It is appropriate when queries and updates are roughly balanced and the implementation must stay short and inspectable. The trade-off: each operation is O(sqrt n), noticeably worse than O(log n), and the block size must be Theta(sqrt n) for the bound to hold. Construction is O(n).

The exported names are `SqrtDecomposition`, `Operation`, `SUM`, `PRODUCT`, `MIN`, and `MAX`. `Operation` is a plain container with `.name`, `.merge`, `.identity`, and `.supports_empty` attributes; build your own for any associative binary operation with an identity element.

## Edges you will hit

* `min` and `max` have no identity over an empty range. Querying an empty range (or an inverted range such as `query(3, 1)`) with `MIN` or `MAX` raises `ValueError`. `SUM` and `PRODUCT` answer an empty range with their identity (0 and 1).
* Out-of-range query indices are clamped to the valid range, matching list slicing — `query(-5, 100)` on a length-3 array is the same as `query(0, 3)`.
* `update(index, value)` with `index == len(self)` appends. `block_size` is fixed at construction and never resized on append; a new trailing block is allocated when the append crosses a block boundary. Any index larger than `len(self)` raises `IndexError`.
* Block indices are `index // block_size`, so `block_size` must be a positive integer; anything else raises `ValueError` at construction.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

