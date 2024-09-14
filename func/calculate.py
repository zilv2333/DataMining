from typing import Union, List

import numpy as np
import pandas as pd
from pandas._typing import AnyArrayLike

Axes = Union[AnyArrayLike, np.ndarray,List]

def EuclideanDistance(x: Axes, y: Axes):
    """
    Calculate the Euclidean distance between two points.
    :param x: point 1
    :param y: point 2
    :return: the Euclidean distance
    """
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def cal_center(x):
    x = np.array(x)
    return np.mean(x, axis=0)


def Ent(probability):
    probability=probability/np.sum(probability,axis=1).reshape(probability.shape[0],-1)
    # print(probability)
    return -np.sum(probability * np.log2(probability), axis=1)


