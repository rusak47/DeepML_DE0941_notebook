
#
# f(x,b) = (x^b + b) * (x^(b-1) + (b - 1)) * (x^(b-2) + (b - 2)) * ... * (x^0 + 0)
def mul(x, b):
    f = 1
    #while b >= 0:
    for i in range(b, -1, -1): # from b to 0
        f *= sum(x, i)
        print(" * ", end="")
        #b -= 1
    return f

def sum(x, b):
    print(f"({x}^{b}+{b} = {pow(x,b)+b})", end="")
    return pow(x,b)+b

def pow(x, b):
    result = 1
    for i in range (b, 0, -1):# from b to 1 # start val; < value; step
        result *= x
    return result

if __name__ == "__main__":
    print("Testing pow:")
    print(f"f(2, 3) = {pow(2, 3)}")
    print(f"f(-2, 3) = {pow(-2, 3)}")
    print(f"f(1, 4) = {pow(1, 4)}")

    print("Testing sum:")
    print(f"f(2, 3) = {sum(2, 3)}")
    print(f"f(-2, 3) = {sum(-2, 3)}")
    print(f"f(1, 4) = {sum(1, 4)}")

    print("Testing mul:")
    print(f"f(2, 3) = {mul(2, 3)}")
    print(f"f(-2, 3) = {mul(-2, 3)}")
    print(f"f(1, 4) = {mul(1, 4)}")
    print(f"f(1, 3) = {mul(1, 3)}")
    print(f"f(1, 2) = {mul(1, 2)}")

