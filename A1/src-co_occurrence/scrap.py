from submission import compute_co_occurrence_matrix


if __name__ == '__main__':
    s = 'all that glitters is not gold'.split()
    s1 = 'all is well that ends well'.split()

    m, i = compute_co_occurrence_matrix([s, s1], 1)
    pass
