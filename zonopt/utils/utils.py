def all_subsets(n):
    """
    Return all binary lists of length n.
    """
    subsets = []
    for i in range(2**n):
        L = list(bin(i)[2:])
        L = [0] * (n - len(L)) + [int(l) for l in L]
        subsets.append(L)
    return subsets
