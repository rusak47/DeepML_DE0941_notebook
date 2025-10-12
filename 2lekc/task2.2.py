import numpy as np
from sympy.polys.numberfields.galois_resolvents import GaloisGroupException


def dot(X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    X_rows = X.shape[0]
    Y_rows = Y.shape[0]
    X_columns = X.shape[1]
    Y_columns = Y.shape[1]
    print(f"shapes: X: {X.shape}, Y: {Y.shape}")

    if X_columns != Y_rows and X_columns != Y_columns:
        raise Exception("dimension mismatch")

    product = np.zeros((X_rows, Y_columns))
    if X_rows > Y_rows:
        product = np.zeros((Y_rows, X_rows))

    print(f"product: {product.shape}")
    # TODO implement algorithm
    for x_ri in range(X_rows):
        print(f"> x_ri {x_ri+1} of {X_rows}:")
        for j_col in range(Y_columns):
            print(f" x_row,x_col: {x_ri}{j_col}={X[x_ri][j_col]}", end="")
            for j_row in range(Y_rows):
                print(f"  j_col,j_row: {j_col}{j_row}={Y[j_row][j_col]}", end="")
                if X_rows > Y_rows:
                    product[j_row][x_ri] += X[x_ri][j_col] * Y[j_row][j_col]
                else:
                    product[x_ri][j_col] += X[x_ri][j_row]*Y[j_row][j_col]
            print("")
        print("------")
    if product.shape[0] == 1:
        product = product.flatten()

    return product

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
print(f"shape: {npDot.shape} ")

dotCB = dot(C,B)
print(dotCB, npDot)
if not (dotCB == npDot).all():
    raise Exception(f"wrong result CB {dotCB}")

npDot = np.dot(B, D)
print(f"shape: {npDot.shape} ")
dotBD = dot(B,D)
print(dotBD, npDot)

if not (dotBD == np.dot(B,D)).all():
    raise Exception(f"wrong result DB {dotBD}")
exit(1)

dimension_mismatch = False
try:
    npDot = np.dot(B, C) #ValueError: shapes (3,2) and (3,) not aligned: 2 (dim 1) != 3 (dim 0)
    print(f"shape: {npDot.shape} ")
except Exception as e:
    dimension_mismatch = True

got_ex = False
try:
    dotBC = dot(B,C)
    print(dotBC, npDot)
except Exception as e:
    got_ex = True

if got_ex != dimension_mismatch:
    raise Exception(f"wrong result for BC")

npDot = np.dot(D, A)
print(f"shape DA: {npDot.shape} ")

dotDA = dot(D,A)
print(dotDA, npDot)
if not (dotDA == npDot).all():
    raise Exception(f"wrong result DA {dotDA}")

