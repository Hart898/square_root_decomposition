import math
import unittest

from square_root_decomposition import SqrtDecomposition, Operation
from square_root_decomposition.core import SUM, PRODUCT, MIN, MAX


class TestConstruction(unittest.TestCase):
    def test_basic_construction_sum(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], SUM)
        self.assertEqual(len(s), 5)
        self.assertEqual(s.data(), [1, 2, 3, 4, 5])

    def test_empty_construction(self):
        s = SqrtDecomposition([], SUM)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.data(), [])

    def test_custom_block_size(self):
        s = SqrtDecomposition(list(range(10)), SUM, block_size=3)
        self.assertEqual(s.query(0, 10), 45)

    def test_invalid_block_size_raises(self):
        with self.assertRaises(ValueError):
            SqrtDecomposition([1, 2, 3], SUM, block_size=0)

    def test_invalid_block_size_negative(self):
        with self.assertRaises(ValueError):
            SqrtDecomposition([1, 2, 3], SUM, block_size=-1)


class TestSumQueries(unittest.TestCase):
    def test_full_range(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], SUM)
        self.assertEqual(s.query(0, 5), 15)

    def test_partial_range(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], SUM)
        self.assertEqual(s.query(1, 4), 9)

    def test_single_element(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], SUM)
        self.assertEqual(s.query(2, 3), 3)

    def test_empty_range_supported_for_sum(self):
        s = SqrtDecomposition([1, 2, 3], SUM)
        self.assertEqual(s.query(1, 1), 0)

    def test_clamping_out_of_range_indices(self):
        s = SqrtDecomposition([1, 2, 3], SUM)
        self.assertEqual(s.query(-5, 100), 6)

    def test_inverted_range_treated_as_empty(self):
        s = SqrtDecomposition([1, 2, 3], SUM)
        self.assertEqual(s.query(2, 0), 0)

    def test_large_array_rounded_block_boundary(self):
        data = list(range(1000))
        s = SqrtDecomposition(data, SUM)
        self.assertEqual(s.query(0, 1000), sum(data))
        self.assertEqual(s.query(100, 700), sum(data[100:700]))
        self.assertEqual(s.query(0, 1), 0)
        self.assertEqual(s.query(999, 1000), 999)


class TestProductQueries(unittest.TestCase):
    def test_product_basic(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], PRODUCT)
        self.assertEqual(s.query(0, 5), 120)

    def test_product_empty(self):
        s = SqrtDecomposition([1, 2, 3], PRODUCT)
        self.assertEqual(s.query(1, 1), 1)

    def test_product_with_zero(self):
        s = SqrtDecomposition([1, 2, 0, 4, 5], PRODUCT)
        self.assertEqual(s.query(0, 5), 0)
        self.assertEqual(s.query(0, 2), 2)


class TestMinMaxQueries(unittest.TestCase):
    def test_min_basic(self):
        s = SqrtDecomposition([3, 1, 4, 1, 5, 9, 2, 6], MIN)
        self.assertEqual(s.query(0, 8), 1)
        self.assertEqual(s.query(2, 5), 1)
        self.assertEqual(s.query(4, 6), 5)

    def test_max_basic(self):
        s = SqrtDecomposition([3, 1, 4, 1, 5, 9, 2, 6], MAX)
        self.assertEqual(s.query(0, 8), 9)
        self.assertEqual(s.query(0, 4), 4)
        self.assertEqual(s.query(5, 8), 9)

    def test_min_empty_raises(self):
        s = SqrtDecomposition([1, 2, 3], MIN)
        with self.assertRaises(ValueError):
            s.query(1, 1)

    def test_max_inverted_range_raises(self):
        s = SqrtDecomposition([1, 2, 3], MAX)
        with self.assertRaises(ValueError):
            s.query(2, 0)

    def test_min_single_block(self):
        s = SqrtDecomposition([5, 3, 7, 2, 8], MIN, block_size=5)
        self.assertEqual(s.query(0, 5), 2)
        self.assertEqual(s.query(2, 4), 2)

    def test_min_single_element_array(self):
        s = SqrtDecomposition([42], MIN)
        self.assertEqual(s.query(0, 1), 42)


class TestUpdates(unittest.TestCase):
    def test_basic_update(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], SUM)
        s.update(2, 10)
        self.assertEqual(s.data(), [1, 2, 10, 4, 5])
        self.assertEqual(s.query(0, 5), 22)
        self.assertEqual(s.query(0, 3), 13)
        self.assertEqual(s.query(3, 5), 9)

    def test_update_preserves_other_queries(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], SUM)
        s.update(5, 100)
        self.assertEqual(s.query(0, 5), 15)
        self.assertEqual(s.query(5, 10), 100 + 7 + 8 + 9 + 10)

    def test_update_then_query_various_ranges(self):
        data = list(range(1, 21))
        s = SqrtDecomposition(data, SUM)
        s.update(10, 100)
        expected = data[:]
        expected[10] = 100
        self.assertEqual(s.query(0, 20), sum(expected))
        self.assertEqual(s.query(8, 12), sum(expected[8:12]))

    def test_update_to_same_value_is_idempotent(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], SUM)
        s.update(2, 3)
        self.assertEqual(s.query(0, 5), 15)

    def test_update_out_of_range(self):
        s = SqrtDecomposition([1, 2, 3], SUM)
        with self.assertRaises(IndexError):
            s.update(5, 99)

    def test_update_negative_index_raises(self):
        s = SqrtDecomposition([1, 2, 3], SUM)
        with self.assertRaises(IndexError):
            s.update(-1, 99)

    def test_append_via_update_at_len(self):
        s = SqrtDecomposition([1, 2, 3], SUM, block_size=2)
        s.update(3, 10)
        self.assertEqual(len(s), 4)
        self.assertEqual(s.data(), [1, 2, 3, 10])
        self.assertEqual(s.query(0, 4), 16)

    def test_append_to_empty_container(self):
        s = SqrtDecomposition([], SUM)
        s.update(0, 7)
        self.assertEqual(len(s), 1)
        self.assertEqual(s.data(), [7])
        self.assertEqual(s.query(0, 1), 7)

    def test_append_multiple_elements(self):
        s = SqrtDecomposition([], SUM, block_size=2)
        for v in [1, 2, 3, 4, 5]:
            s.update(len(s), v)
        self.assertEqual(s.data(), [1, 2, 3, 4, 5])
        self.assertEqual(s.query(0, 5), 15)

    def test_append_spans_block_boundary(self):
        s = SqrtDecomposition([1, 2], SUM, block_size=2)
        s.update(2, 3)
        self.assertEqual(s.data(), [1, 2, 3])
        self.assertEqual(s.query(0, 3), 6)
        self.assertEqual(s.query(2, 3), 3)

    def test_update_last_element(self):
        s = SqrtDecomposition([1, 2, 3, 4, 5], MAX)
        s.update(4, 100)
        self.assertEqual(s.query(0, 5), 100)


class TestCustomOperation(unittest.TestCase):
    def test_custom_xor_operation(self):
        XOR = Operation("xor", lambda a, b: a ^ b, 0, supports_empty=True)
        s = SqrtDecomposition([1, 2, 3, 4, 5, 6, 7], XOR, block_size=3)
        expected = 0
        for v in [1, 2, 3, 4, 5, 6, 7]:
            expected ^= v
        self.assertEqual(s.query(0, 7), expected)
        self.assertEqual(s.query(1, 4), 2 ^ 3 ^ 4)
        s.update(3, 0)
        self.assertEqual(s.query(0, 7), 1 ^ 2 ^ 3 ^ 0 ^ 5 ^ 6 ^ 7)


class TestRepresentation(unittest.TestCase):
    def test_repr_mentions_op_and_block_size(self):
        s = SqrtDecomposition([1, 2, 3], SUM, block_size=2)
        r = repr(s)
        self.assertIn("sum", r)
        self.assertIn("2", r)

    def test_operation_repr(self):
        self.assertIn("sum", repr(SUM))


class TestFloatData(unittest.TestCase):
    def test_floats_with_sum(self):
        data = [0.1, 0.2, 0.3, 0.4]
        s = SqrtDecomposition(data, SUM)
        self.assertAlmostEqual(s.query(0, 4), sum(data))

    def test_floats_with_min(self):
        data = [3.5, 1.2, 4.7, 0.9, 2.1]
        s = SqrtDecomposition(data, MIN)
        self.assertEqual(s.query(0, 5), 0.9)
        self.assertEqual(s.query(1, 4), 0.9)


if __name__ == "__main__":
    unittest.main()
