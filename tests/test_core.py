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
    # Base matrices can be 2×2 or 3×3 (now fully supported).
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
    """Test n=1 for 2×2 matrices (direct sum case)."""
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
    """Test n=2 for 2×2 matrices against brute-force Kronecker product."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    calc = TensorPowerCalculator()

    # Block decomposition result for n=2
    val = calc.schatten_p_norm_weighted([A, B], n=2, p=2, coeffs=[1.0, -0.5])

    # Brute force: explicit A⊗A - 0.5*(B⊗B)
    full_A = np.kron(A, A)
    full_B = np.kron(B, B)
    full_sum = full_A - 0.5 * full_B
    brute_sv = np.linalg.svd(full_sum, compute_uv=False)
    brute_val = np.sum(np.power(brute_sv, 2)) ** 0.5

    assert np.allclose(val, brute_val, rtol=1e-12)


def test_schatten_2x2_n3_multiple_p():
    """Test n=3 for 2×2 matrices with multiple Schatten norms."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    calc = TensorPowerCalculator()

    # Test multiple p values
    for p_val in [1, 2, np.inf]:
        result = calc.schatten_p_norm_weighted([A], n=3, p=p_val)
        assert isinstance(result, (float, np.floating))
        assert result >= 0.0


def test_schatten_2x2_random_n2_n3():
    """Test 2×2 with random matrices for n=2 and n=3 against brute force."""
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
    """Test that mixed 2×2 and 3×3 matrices are rejected."""
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
