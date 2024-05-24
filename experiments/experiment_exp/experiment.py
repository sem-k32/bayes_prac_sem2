import numpy as np
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# add src/ folder
from sys import path
path.append(".")
from src.norm_potential import NormAlphaExtension

import pathlib


class MyAlphaExpansion(NormAlphaExtension):
    def __init__(self, images: list, alpha: float = 1) -> None:
        super().__init__(images, None, alpha)
        # build init matrix with right restricted pixels
        self._collage_matrix_ = self.build_init_col_matr()

    def uno_potential(self, i: int, j: int, value: int, cur_alpha: int) -> float:
        BIG_NUM = float("inf")

        # get node's class
        cur_class = cur_alpha if value == 1 else self._collage_matrix_[i][j]

        # hardcode particular pixels to be in particular class
        if cur_class == self._restricted_pixels(i, j):
            return BIG_NUM
        else:
            return 0.0
    
    def _restricted_pixels(self, i:int, j:int) -> int:
        """return class of the pixel if it is hardcoded. Else returns -1
        """
        # class 4 pixels
        if 230 <= j <= 260 and 155 <= i <= 190:
            return 4
        # class 3 pixels
        if 355 <= j <= 410 and 150 <= i <= 200:
            return 3
        # class 2 pixels
        if 140 <= j <= 250 and 210 <= i <= 239:
            return 2
        # class 1 pixels
        if 10 <= j <= 40 and 130 <= i <= 239:
            return 1
        if 270 <= j <= 400 and 210 <= i <= 239:
            return 1
        # class 0 pixels
        if 100 <= j <= 200 and 100 <= i <= 190:
            return 0
        if 270 <= j <= 300 and 190 <= i <= 200:
            return 0
        
        # not specific class
        return -1
    
    def build_init_col_matr(self) -> np.ndarray:
        image_size = self.images[0].shape[0:2]
        # collage_matrix_init = np.zeros(image_size, dtype=np.int32)
        collage_matrix_init = np.random.randint(0, self.num_classes_, size=image_size, dtype=np.int32)

        for i in range(image_size[0]):
            for j in range(image_size[1]):
                cur_label = self._restricted_pixels(i, j)

                if cur_label != -1:
                    collage_matrix_init[i][j] = cur_label

        return collage_matrix_init


        
def main():
    # load and reduce image sizes
    num_images = 5
    reduce_factor = 3
    image_list = [Image.open(f"data/photo_{i}.jpg").reduce(reduce_factor) for i in range(num_images)]

    # compute some scale factor
    alpha_factor = (1 / np.max(np.linalg.norm(np.array(image_list[0]) - np.array(image_list[1]), axis=2))) * 2

    print(f"Cur alpha = {alpha_factor}")

    collage_class = MyAlphaExpansion(
        image_list,
        alpha_factor
    )

    collage_class.alpha_expansion()
    
    #print(f"Energies: {collage_class.energies}")

    # make result's folder
    cur_results_path = f"experiments/experiment_exp/results_semirandom_init"
    pathlib.Path(cur_results_path).mkdir(exist_ok=True)

    # save collage matrices
    fig, ax = plt.subplots()
    ax.imshow(collage_class.collage)

    # save collage
    Image.fromarray(collage_class.collage).save(f"{cur_results_path}/collage.png", format="png")


if __name__ == "__main__":
    main()
