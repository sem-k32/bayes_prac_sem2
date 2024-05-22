from typing import List
import numpy as np
from PIL import Image
from .base_alpha_expansion import BaseAlphaExpansion

class NormAlphaExtension(BaseAlphaExpansion):
    """ class defines duel potentials as metrics so that they satisfy triangle inequality
    """
    def __init__(self, images: List[Any], init_collage_matrix: np.ndarray = None) -> None:
        super().__init__(images, init_collage_matrix)

    def duel_potential(self, i: int, j: int, n: int, k: int, value_1: int, value_2: int) -> float:
        if value_1 == value_2:
            return 0
        else:
            temp1 = self.images_[value_1][i][j] 
            temp2 = self.images_[value_2][n][k]
            temp3 = np.linalg.norm(self.images_[value_1][i][j] - self.images_[value_2][n][k])

            return np.exp(self.lambd_ * np.linalg.norm(self.images_[value_1][i][j] - self.images_[value_2][n][k]))
