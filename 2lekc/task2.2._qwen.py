import numpy as np


def dot(X, Y):
    """
    Educational implementation of dot product for 1D and 2D arrays.
    Mimics np.dot behavior with detailed logging and error messages.

    Supports:
      - vector · vector     → scalar
      - matrix · vector     → vector
      - vector · matrix     → vector
      - matrix · matrix     → matrix

    Parameters:
        X: 1D or 2D array-like
        Y: 1D or 2D array-like

    Returns:
        Result of dot product (scalar, 1D, or 2D numpy array)

    Raises:
        ValueError: if inner dimensions do not match
    """
    # Convert inputs to numpy arrays for consistency
    X = np.array(X)
    Y = np.array(Y)

    # Log shapes for educational purposes
    print(f"Input shapes: X.shape = {X.shape}, Y.shape = {Y.shape}")

    # Handle 1D-1D: vector dot product → scalar
    if X.ndim == 1 and Y.ndim == 1:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Vector dot product requires same length. Got {X.shape[0]} and {Y.shape[0]}")
        print("→ Case: 1D · 1D → scalar")
        result = 0.0
        for i in range(len(X)):
            result += X[i] * Y[i]
            print(f"  step {i + 1}: {X[i]} * {Y[i]} = {X[i] * Y[i]} → cumulative: {result}")
        print(f"→ Final scalar result: {result}")
        return result

    # Handle 2D-1D: matrix · vector → vector (Y treated as column vector)
    elif X.ndim == 2 and Y.ndim == 1:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                f"Matrix-vector multiplication: X columns ({X.shape[1]}) must match Y length ({Y.shape[0]})")
        print("→ Case: 2D · 1D → 1D (matrix times column vector)")
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):  # for each row in matrix
            for k in range(X.shape[1]):  # for each element in row
                result[i] += X[i, k] * Y[k]
                print(f"  result[{i}] += X[{i},{k}] * Y[{k}] = {X[i, k]} * {Y[k]} → {result[i]}")
        print(f"→ Final vector result: {result}")
        return result

    # Handle 1D-2D: vector · matrix → vector (X treated as row vector)
    elif X.ndim == 1 and Y.ndim == 2:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Vector-matrix multiplication: X length ({X.shape[0]}) must match Y rows ({Y.shape[0]})")
        print("→ Case: 1D · 2D → 1D (row vector times matrix)")
        result = np.zeros(Y.shape[1])
        for j in range(Y.shape[1]):  # for each column in matrix
            for k in range(Y.shape[0]):  # for each row (dot with vector)
                result[j] += X[k] * Y[k, j]
                print(f"  result[{j}] += X[{k}] * Y[{k},{j}] = {X[k]} * {Y[k, j]} → {result[j]}")
        print(f"→ Final vector result: {result}")
        return result

    # Handle 2D-2D: matrix · matrix
    elif X.ndim == 2 and Y.ndim == 2:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(f"Matrix multiplication: X columns ({X.shape[1]}) must match Y rows ({Y.shape[0]})")
        print("→ Case: 2D · 2D → 2D")
        result = np.zeros((X.shape[0], Y.shape[1]))
        for i in range(X.shape[0]):  # for each row in X
            for j in range(Y.shape[1]):  # for each column in Y
                for k in range(X.shape[1]):  # sum over shared dimension
                    result[i, j] += X[i, k] * Y[k, j]
                    print(f"  result[{i},{j}] += X[{i},{k}] * Y[{k},{j}] = {X[i, k]} * {Y[k, j]} → {result[i, j]}")
                print(f"  → row {i}, col {j} done: {result[i, j]}")
        print(f"→ Final matrix result:\n{result}")
        return result

    else:
        raise ValueError("Only 1D and 2D arrays supported")

def test_dot_product():
    print("="*60)
    print("🧪 COMPREHENSIVE DOT PRODUCT TESTS")
    print("="*60)

    # Test 1: 1D · 1D → scalar
    print("\n🔹 TEST 1: Vector Dot Product (1D · 1D → scalar)")
    a = [1, 2, 3]
    b = [4, 5, 6]
    result = dot(a, b)
    expected = np.dot(a, b)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ PASSED\n")

    # Test 2: 2D · 1D → 1D (matrix times column vector)
    print("\n🔹 TEST 2: Matrix · Vector (2D · 1D → 1D)")
    A = [[1, 2, 3],
         [4, 5, 6]]
    b = [1, 0, 1]
    result = dot(A, b)
    expected = np.dot(A, b)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✅ PASSED\n")

    # Test 3: 1D · 2D → 1D (row vector times matrix)
    print("\n🔹 TEST 3: Vector · Matrix (1D · 2D → 1D)")
    a = [1, 2]
    B = [[1, 2, 3],
         [4, 5, 6]]
    result = dot(a, B)
    expected = np.dot(a, B)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✅ PASSED\n")

    # Test 4: 2D · 2D → 2D
    print("\n🔹 TEST 4: Matrix · Matrix (2D · 2D → 2D)")
    A = [[1, 2],
         [3, 4]]
    B = [[5, 6],
         [7, 8]]
    result = dot(A, B)
    expected = np.dot(A, B)
    assert np.allclose(result, expected), f"Expected\n{expected}\ngot\n{result}"
    print("✅ PASSED\n")

    # Test 5: Dimension mismatch (should raise error)
    print("\n🔹 TEST 5: Dimension Mismatch (should raise ValueError)")
    A = [[1, 2, 3]]   # shape (1,3)
    b = [1, 2]        # shape (2,)
    error_caught = False
    try:
        dot(A, b)  # columns=3, vector len=2 → mismatch
    except ValueError as e:
        print(f"→ Correctly caught error: {e}")
        error_caught = True
    assert error_caught, "Should have raised ValueError for dimension mismatch"
    print("✅ PASSED\n")

    # Test 6: Non-supported dimensions
    print("\n🔹 TEST 6: 3D Array (should raise error)")
    X = np.zeros((2,2,2))
    Y = [1, 2]
    error_caught = False
    try:
        dot(X, Y)
    except ValueError as e:
        print(f"→ Correctly caught error: {e}")
        error_caught = True
    assert error_caught, "Should have raised error for 3D input"
    print("✅ PASSED\n")

    # Test 7: Edge case — scalar-like result from (1,1) matrix
    print("\n🔹 TEST 7: (1,1) Matrix Result")
    A = [[2]]      # 2D
    B = [[3]]      # 2D
    result = dot(A, B)
    expected = np.dot(A, B)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    assert result.shape == (1,1), "Should return 2D (1,1) matrix, not scalar"
    print(f"→ Result shape: {result.shape}, value: {result}")
    print("✅ PASSED\n")

    # Test 8: Vector · Vector → scalar, not array
    print("\n🔹 TEST 8: Vector·Vector returns scalar (not array)")
    a = [1, 1]
    b = [2, 3]
    result = dot(a, b)
    assert isinstance(result, (int, float)), "Vector·vector should return scalar"
    expected = np.dot(a, b)
    assert result == expected
    print(f"→ Result: {result} (type: {type(result).__name__})")
    print("✅ PASSED\n")

    print("🎉 ALL TESTS PASSED!")

    D = np.array([1, 2])
    A = np.array([
        [1, 3, 6],
        [5, 2, 8]
    ])

    C = np.array([1, 2, 3])
    B = np.array([
        [1, 3],
        [5, 2],
        [6, 9]
    ])

    npDot = np.dot(C, B)
    #print(f"shape: {npDot.shape} ")

    dotCB = dot(C, B)
    print(dotCB, npDot)
    if not (dotCB == npDot).all():
        raise Exception(f"wrong result CB {dotCB}")

    npDot = np.dot(B, D)
    #print(f"shape: {npDot.shape} ")
    dotBD = dot(B, D)
    print(dotBD, npDot)

    if not (dotBD == np.dot(B, D)).all():
        raise Exception(f"wrong result DB {dotBD}")

    dimension_mismatch = False
    try:
        npDot = np.dot(B, C)  # ValueError: shapes (3,2) and (3,) not aligned: 2 (dim 1) != 3 (dim 0)
        print(f"shape: {npDot.shape} ")
    except Exception as e:
        dimension_mismatch = True

    got_ex = False
    try:
        dotBC = dot(B, C)
        print(dotBC, npDot)
    except Exception as e:
        got_ex = True
        print(f"Correctly cauht exception DC {e}")

    if got_ex != dimension_mismatch:
        raise Exception(f"wrong result for BC")

    npDot = np.dot(D, A)
    print(f"shape DA: {npDot.shape} ")

    dotDA = dot(D, A)
    print(dotDA, npDot)
    if not (dotDA == npDot).all():
        raise Exception(f"wrong result DA {dotDA}")
test_dot_product()