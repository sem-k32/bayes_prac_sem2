import numpy as np
from PIL import Image
from .base_graphcut import BaseGraphCut

class ExpGraphCut(BaseGraphCut):
    def __init__(self, image_1: Image, image_2: Image, lambd: float) -> None:
        """
        Args:
            lambd (float): constant to scale in exp potential
        """
        super().__init__(image_1, image_2)

        self.lambd_ = lambd

    def get_neighbours(cls, i: int, j: int) -> list:
        im_height, im_width = cls.images_[0].shape[0:2]

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
            temp1 = self.images_[value_1][i][j] 
            temp2 = self.images_[value_2][n][k]
            temp3 = np.linalg.norm(self.images_[value_1][i][j] - self.images_[value_2][n][k])

            return np.exp(self.lambd_ * np.linalg.norm(self.images_[value_1][i][j] - self.images_[value_2][n][k]))
