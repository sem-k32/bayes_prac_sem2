from typing import List
import numpy as np
from PIL import Image
from .base_alpha_expansion import BaseAlphaExpansion

class NormAlphaExtension(BaseAlphaExpansion):
    """ class defines duel potentials as metrics so that they satisfy triangle inequality
    """
    def __init__(self, image: Image, num_classes: int, init_seg_matrix: np.ndarray = None, lambd: float = 1) -> None:
        """
        Args:
            lambd (float): coeff for norm function
        """
        super().__init__(image, num_classes, init_seg_matrix)

        self.lambd_ = lambd

    def duel_potential(self, i: int, j: int, n: int, k: int, value_1: int, value_2: int, cur_alpha: int) -> float:
        """ implements potentials as metric functions based on vector norm
        """
        # get current classes of nodes
        class_1 = self._seg_matrix_[i][j]
        class_2 = self._seg_matrix_[n][k]

        # get classes according to indicators value_1 and value_2
        class_left = class_1 if value_1 == 0 else cur_alpha
        class_right = class_2 if value_2 == 0 else cur_alpha
        
        pixel_left = self.image_[i][j]
        pixel_right = self.image_[n][k]

        return (class_left != class_right) * self.lambd_ * np.linalg.norm(pixel_left - pixel_right, ord=1)

            
