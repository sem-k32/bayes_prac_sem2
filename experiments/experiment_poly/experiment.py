import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# add src/ folder
from sys import path
path.append(".")
from src.exp_potential import ExpGraphCut
from src.poly_potential import PolyGraphCut

import pathlib

class MyGhraphCut(PolyGraphCut):
    """final graph cut class with uno potential defined regarding images to be used
            (photo_1, photo_2)
    """
    def uno_potential(self, i: int, j: int, value: int) -> float:
        """hardcode some pixels to be in particular class
        """
        # image 0 pixels to include for sure
        if (200 <= i <= 250) and (150 <= j <= 250) and value == 1:
            return 1e8
        if (290 <= i <= 310) and (410 <= j <= 430) and value == 1:
            return 1e8

        # image 1 pixels to include for sure
        if (325 <= i <= 350) and (400 <= j <= 600) and value == 0:
            return 1e8
        if (260 <= i <= 330) and (10 <= j <= 50) and value == 0:
            return 1e8
        
        # otherwise return constant
        return 0
        

def main():
    # load and reduce image sizes
    reduce_factor = 2
    image_1 = Image.open("data/photo_1.jpg").reduce(reduce_factor)
    image_2 = Image.open("data/photo_2.jpg").reduce(reduce_factor)

    #alpha_array = [0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2]
    alpha_array = [1.1]

    for alpha in alpha_array:
        print(f"Cur alpha = {alpha}")

        graph_cut = MyGhraphCut(
            image_1,
            image_2,
            alpha=alpha
        )

        # obtain collage matrix and collage
        print("Final energy:", graph_cut.compose_collage())

        # make result's folder
        cur_results_path = f"experiments/experiment_poly/results_alpha_{alpha:.2f}"
        pathlib.Path(cur_results_path).mkdir(exist_ok=True)

        # save collage matrix
        fig, ax = plt.subplots()
        ax.imshow(graph_cut.collage_matrix, cmap="binary")
        # legend
        img_0_patch = mpatches.Patch(color='white', label='img_0')
        img_1_patch = mpatches.Patch(color='black', label='img_1')
        ax.legend(handles=[img_0_patch, img_1_patch])

        fig.savefig(f"{cur_results_path}/collage_matrix.png", format="png")

        # save collage
        Image.fromarray(graph_cut.collage).save(f"{cur_results_path}/collage.png", format="png")


if __name__ == "__main__":
    main()
