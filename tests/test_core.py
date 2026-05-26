import numpy as np
import pytest

from tensorpow import TensorPowerCalculator, solve_su3_pieri


def test_basic():
    # ILP solutions should exist for several n values; spot-check a couple
    for n in (2, 3, 5):
        res = solve_su3_pieri(n)
        assert res["status"] == "Optimal"
        assert isinstance(res["solution"], list)


def test_schatten_n1():
    # for n=1 the calculator should just compute the ordinary p-norm
    A = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    B = np.array([[0.5, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -0.5]])
    calc = TensorPowerCalculator()
    # weighted case with coefficients (p=2)
    val1 = calc.schatten_p_norm_weighted([A, B], n=1, p=2, coeffs=[1.0, 1.0])
    direct = np.linalg.svd(A + B, compute_uv=False)
    direct_val = np.sum(np.power(direct, 2)) ** 0.5
    assert np.allclose(val1, direct_val)

    # p = infinity should give max singular value of the weighted sum
    inf_val = calc.schatten_p_norm_weighted([A, B], n=1, p=np.inf, coeffs=[1.0, 1.0])
    expected_inf = np.max(np.linalg.svd(A + B, compute_uv=False))
    assert inf_val == pytest.approx(expected_inf)

    # also accept a single array without wrapping
    single = calc.schatten_p_norm_weighted(A, n=1, p=2)
    wrapped = calc.schatten_p_norm_weighted([A], n=1, p=2)
    assert single == pytest.approx(wrapped)


def test_schatten_random_nk():
    # compare weighted norm for several powers (2 and 3)
    rng = np.random.default_rng(0)
    for power in (2, 3):
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        calc = TensorPowerCalculator()
        val = calc.schatten_p_norm_weighted([A, B], n=power, p=2, coeffs=[1, -1])
        # brute force using explicit kronecker product
        fullA = A
        for _ in range(power - 1):
            fullA = np.kron(fullA, A)
        fullB = B
        for _ in range(power - 1):
            fullB = np.kron(fullB, B)
        full = fullA - fullB
        brute = np.linalg.svd(full, compute_uv=False)
        brute_val = np.sum(np.power(brute, 2)) ** 0.5
        assert np.allclose(val, brute_val)


def test_invalid_dimension_and_size():
    # Powers n must be a positive integer.
    # Base matrices can be 2x2 or 3x3 (now fully supported).
    calc = TensorPowerCalculator()

    # invalid powers raise
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([np.ones((3, 3))], n=0)
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([np.ones((3, 3))], n=-1)

    # unsupported sizes (not 2 or 3) are rejected
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([np.ones((4, 4))], n=3)
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([np.ones((5, 5))], n=3)

    # non-square array rejects
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([np.ones((3, 2))], n=3)
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([np.ones((2, 3))], n=2)

    # constructor misuse should surface with a clear message
    with pytest.raises(TypeError):
        TensorPowerCalculator(np.eye(3))


def test_schatten_n1_2x2():
    """Test n=1 for 2x2 matrices (direct sum case)."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    calc = TensorPowerCalculator()

    # weighted case with coefficients (p=2)
    val = calc.schatten_p_norm_weighted([A, B], n=1, p=2, coeffs=[1.0, -0.5])
    direct_sum = A - 0.5 * B
    direct_sv = np.linalg.svd(direct_sum, compute_uv=False)
    direct_val = np.sum(np.power(direct_sv, 2)) ** 0.5
    assert np.allclose(val, direct_val)

    # p=infinity should give max singular value
    inf_val = calc.schatten_p_norm_weighted([A, B], n=1, p=np.inf, coeffs=[1.0, -0.5])
    expected_inf = np.max(np.linalg.svd(direct_sum, compute_uv=False))
    assert inf_val == pytest.approx(expected_inf)


def test_schatten_2x2_n2_brute_force():
    """Test n=2 for 2x2 matrices against brute-force Kronecker product."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    calc = TensorPowerCalculator()

    # Block decomposition result for n=2
    val = calc.schatten_p_norm_weighted([A, B], n=2, p=2, coeffs=[1.0, -0.5])

    # Brute force: explicit A(kron)A - 0.5*(B(kron)B)
    full_A = np.kron(A, A)
    full_B = np.kron(B, B)
    full_sum = full_A - 0.5 * full_B
    brute_sv = np.linalg.svd(full_sum, compute_uv=False)
    brute_val = np.sum(np.power(brute_sv, 2)) ** 0.5

    assert np.allclose(val, brute_val, rtol=1e-12)


def test_schatten_2x2_n3_multiple_p():
    """Test n=3 for 2x2 matrices with multiple Schatten norms."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    calc = TensorPowerCalculator()

    # Test multiple p values
    for p_val in [1, 2, np.inf]:
        result = calc.schatten_p_norm_weighted([A], n=3, p=p_val)
        assert isinstance(result, (float, np.floating))
        assert result >= 0.0


def test_schatten_2x2_random_n2_n3():
    """Test 2x2 with random matrices for n=2 and n=3 against brute force."""
    rng = np.random.default_rng(42)

    for power in (2, 3):
        A = rng.standard_normal((2, 2))
        B = rng.standard_normal((2, 2))
        calc = TensorPowerCalculator()

        val = calc.schatten_p_norm_weighted([A, B], n=power, p=2, coeffs=[1, -1])

        # Brute force using explicit kronecker product
        full_A = A
        for _ in range(power - 1):
            full_A = np.kron(full_A, A)
        full_B = B
        for _ in range(power - 1):
            full_B = np.kron(full_B, B)
        full = full_A - full_B
        brute_sv = np.linalg.svd(full, compute_uv=False)
        brute_val = np.sum(np.power(brute_sv, 2)) ** 0.5

        assert np.allclose(val, brute_val, rtol=1e-12)


def test_mixed_dimensions_rejected():
    """Test that mixed 2x2 and 3x3 matrices are rejected."""
    calc = TensorPowerCalculator()

    A2 = np.eye(2)
    A3 = np.eye(3)

    # Should reject mixed dimensions for any n
    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([A2, A3], n=1)

    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([A2, A3], n=2)

    with pytest.raises(ValueError):
        calc.schatten_p_norm_weighted([A3, A2], n=3)


# ---------------------------------------------------------------------------
# Block Decomposition Tests
# ---------------------------------------------------------------------------


def test_block_decomposition_n1():
    """Test block decomposition returns the matrix itself for n=1."""
    calc = TensorPowerCalculator()

    # 2x2
    A2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    res2 = calc.block_decomposition(A2, n=1)
    assert len(res2) == 1
    assert res2[0][0] == 1.0
    assert np.allclose(res2[0][1], A2)

    # 3x3
    A3 = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    res3 = calc.block_decomposition(A3, n=1)
    assert len(res3) == 1
    assert res3[0][0] == 1.0
    assert np.allclose(res3[0][1], A3)


def test_block_decomposition_2x2_n5():
    """Test 2x2 block decomposition spectral equivalence for n=5."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((2, 2))
    n = 5
    calc = TensorPowerCalculator()

    blocks = calc.block_decomposition(M, n=n)

    # 1. Test dimension conservation (sum of block_dim * mult == 2^n)
    total_dim = sum(mult * block.shape[0] for mult, block in blocks)
    assert total_dim == 2**n

    # 2. Reconstruct brute force M^⊗n
    M_full = M
    for _ in range(n - 1):
        M_full = np.kron(M_full, M)
    expected_svs = np.linalg.svd(M_full, compute_uv=False)

    # 3. Collect and flatten singular values from all blocks
    block_svs = []
    for mult, block in blocks:
        svs = np.linalg.svd(block, compute_uv=False)
        # Duplicate the singular values according to the block's multiplicity
        for _ in range(int(mult)):
            block_svs.extend(svs)

    # Sort descending to align with np.linalg.svd defaults
    block_svs = np.sort(block_svs)[::-1]

    # 4. Assert spectral equivalence
    assert np.allclose(expected_svs, block_svs, rtol=1e-11)


def test_block_decomposition_3x3_n5():
    """Test 3x3 SU(3) virtual block decomposition spectral equivalence for n=5."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((3, 3))
    n = 5
    calc = TensorPowerCalculator()

    blocks = calc.block_decomposition(M, n=n)

    # 1. Test virtual dimension conservation (sum of block_dim * mult == 3^n)
    # Note: Because of Jacobi-Trudi, mult can be negative!
    total_dim = sum(mult * block.shape[0] for mult, block in blocks)
    assert total_dim == 3**n

    # 2. Reconstruct brute force M^⊗n
    M_full = M
    for _ in range(n - 1):
        M_full = np.kron(M_full, M)

    # 3. Get brute force singular values
    expected_svs = np.linalg.svd(M_full, compute_uv=False)

    # 4. Because the decomposition is virtual (has negative multiplicities),
    # we cannot map singular values 1-to-1. Instead, we verify that the
    # algebraic sum of their p-th powers is identical.

    # Check for p=2 (Frobenius-type norm)
    expected_p2 = np.sum(np.power(expected_svs, 2))
    block_p2 = 0.0
    for mult, block in blocks:
        svs = np.linalg.svd(block, compute_uv=False)
        block_p2 += mult * np.sum(np.power(svs, 2))

    assert np.allclose(expected_p2, block_p2, rtol=1e-11)

    # Check for p=4 to ensure higher-order moment equivalence
    expected_p4 = np.sum(np.power(expected_svs, 4))
    block_p4 = 0.0
    for mult, block in blocks:
        svs = np.linalg.svd(block, compute_uv=False)
        block_p4 += mult * np.sum(np.power(svs, 4))

    assert np.allclose(expected_p4, block_p4, rtol=1e-11)
