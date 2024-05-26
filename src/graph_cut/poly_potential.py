import numpy as np
from PIL import Image
from .base_graphcut import BaseGraphCut

class PolyGraphCut(BaseGraphCut):
    """implements duel potential as potts model + polynome of pixels distance
    """
    def __init__(self, image: Image, lambd: float, beta: float) -> None:
        """
        Args:
            lambd (float): degree of the poly
            beta (float): constant to scale in poly potential
        """
        super().__init__(image)

        self.lambd_ = lambd
        self.beta_ = beta

    def get_neighbours(cls, i: int, j: int) -> list:
        im_height, im_width = cls.image_.shape[0:2]

        neighb_list = []
        if i != 0:
            neighb_list.append((i - 1, j))
        if i != im_height - 1:
            neighb_list.append((i + 1, j))
        if j != 0:
            neighb_list.append((i, j - 1))
        if j != im_width - 1:
            neighb_list.append((i, j + 1))

        return neighb_list

    def duel_potential(self, i: int, j: int, n: int, k: int, value_1: int, value_2: int) -> float:
        if value_1 == value_2:
            return 0
        else:
            return (0.01 + self.beta_ * np.linalg.norm(self.image_[i][j] - self.image_[n][k], ord=1)) ** self.lambd_
