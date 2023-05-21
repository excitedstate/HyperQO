import sys

sys.path.append("..")


def test_knn():
    from src.knn import KNN

    target_data = (2, 2, 2)

    knn = KNN(2)
    # # case 1
    knn.insert_values(((1, 2, 3), 1),
                      ((2, 3, 4), 2),
                      ((3, 4, 5), 3),
                      ((4, 5, 6), 4))
    assert_res = [(0, 1), (1, 2)]
    assert knn(target_data) == assert_res, "case 1 failed!"
    print("\n[SUCCESS] KNN case 1 passed!")

    # # case 2
    knn.insert_values(((2, 2, 3), 5))
    assert_res = [(0, 1), (0, 5)]
    assert knn(target_data) == assert_res, "case 2 failed!"
    print("[SUCCESS] KNN case 2 passed!")

    # # case 3
    for _ in range(100):
        assert knn.k_neighbours_sample(target_data) in [1, 5], "case 3 failed!"
    print("[SUCCESS] KNN case 3 passed!")


if __name__ == "__main__":
    test_knn()
