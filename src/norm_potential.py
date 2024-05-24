from typing import List
import numpy as np
from PIL import Image
from .base_alpha_expansion import BaseAlphaExpansion

class NormAlphaExtension(BaseAlphaExpansion):
    """ class defines duel potentials as metrics so that they satisfy triangle inequality
    """
    def __init__(self, images: list, init_collage_matrix: np.ndarray = None, alpha: float = 1) -> None:
        """
        Args:
            alpha (float): coeff for norm function
        """
        super().__init__(images, init_collage_matrix)

        self.alpha_ = alpha

    def duel_potential(self, i: int, j: int, n: int, k: int, value_1: int, value_2: int, cur_alpha: int) -> float:
        """ implements potentials as metric functions based on vector norm
        """
        # get current classes of nodes
        class_1 = self._collage_matrix_[i][j]
        class_2 = self._collage_matrix_[n][k]

        # get classes according to indicators value_1 and value_2
        class_left = class_1 if value_1 == 0 else cur_alpha
        class_right = class_2 if value_2 == 0 else cur_alpha
        
        pixel_left = self.images_[class_left][i][j]
        pixel_right = self.images_[class_right][n][k]

        return (class_left != class_right) * self.alpha_ * np.linalg.norm(pixel_left - pixel_right)

            
