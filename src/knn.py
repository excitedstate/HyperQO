import random
import typing
import collections

import numpy


class KNN:
    __data_repr = collections.namedtuple("KNNData", ["point", "id"])
    __infer_result_repr = collections.namedtuple("KNNInferenceResult", ["distance", "id"])

    def __init__(self, k: int,
                 max_distance=0.05 ** 2 + 0.05 ** 2 + 0.05 ** 2 + 10,
                 max_v=4,
                 distance_judge: typing.Optional[typing.Callable[[typing.Any, typing.Any], float]] = None):
        """
            KNN algorithm, a classic classification algorithm.
        @param k: parameter k in KNN, the number of neighbours to be considered.
        @param max_distance: distance threshold, if the distance between the target and the kth neighbour is larger than
        @param max_v: the value to be returned if the distance is larger than max_distance.
        @param distance_judge:
        """
        self.k = k
        self.data = list()
        self.max_distance = max_distance
        self.max_v = max_v
        self.distance = distance_judge if distance_judge is not None else self.distance

    def insert_values(self, *data: typing.Iterable):
        """
            Insert data into the KNN.
        @param data: example: ( (1,2,3) ,1), ... [ point, id ]
        @return:
        """
        self.data.extend(map(lambda x: self.__data_repr(*x), data))

    def k_neighbours_sample(self, v):
        """
            Get the k neighbours of the target point, **but return a random one**.
        @param v:
        @return:
        """
        res = self(v)

        # # res null or min(distance(res)) > max_distance
        if len(res) == 0 or res[-1][0] > self.max_distance:
            return self.max_v

        # # return a random one.
        return random.sample(res, 1)[0][1]

    @staticmethod
    def distance(v1, v2) -> float:
        """
            Calculate the distance between two vectors. but just second dimension.
        @param v1:
        @param v2:
        @return:
        """

        assert len(v1) == len(v2), "The length of two vectors must be the same."
        return sum([abs(v[0] - v[1]) for v in zip(v1[1:2], v2[1:2])])

    def __call__(self, v):
        if len(self.data) < self.k:
            return list()

        # # translate (x, id) -> (distance(x, v), id)
        # # sort by distance, and get the first k elements.

        chosen_data = sorted(
            map(lambda x: (
                self.distance(v, x[0]), x[1]),
                self.data), key=lambda x: x[0])[:self.k]
        return list(map(lambda x: self.__infer_result_repr(*x), chosen_data))


if __name__ == "__main__":
    pass
