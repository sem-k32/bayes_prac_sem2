import numpy as np
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# add src/ folder
from sys import path
path.append(".")
from src.exp_potential import ExpGraphCut
from src.poly_potential import PolyGraphCut

import pathlib


class MyGhraphCut(ExpGraphCut):
    """final graph cut class with uno potential defined regarding images to be used
            (photo_1, photo_2)
    """
    def uno_potential(self, i: int, j: int, value: int) -> float:
        # image 0 pixels to include for sure
        # if (350 <= i <= 500) and (320 <= j <= 500) and value == 1:
        #     return np.inf
        # if (550 <= i <= 590) and (820 <= j <= 890) and value == 1:
        #     return np.inf

        # # image 1 pixels to include for sure
        # if (700 <= i <= 720) and (800 <= j <= 1280) and value == 0:
        #     return np.inf
        # if (500 <= i <= 550) and (50 <= j <= 100) and value == 0:
        #     return np.inf

        BIG_NUM = float("inf")

        # image 0 pixels to include for sure
        if (110 <= i <= 200) and (90 <= j <= 170) and value != 0:
            return BIG_NUM
        if (185 <= i <= 200) and (280 <= j <= 300) and value != 0:
            return BIG_NUM

        # image 1 pixels to include for sure
        if (130 <= i <= 220) and (10 <= j <= 50) and value != 1:
            return BIG_NUM
        if (215 <= i <= 239) and (290 <= j <= 400) and value != 1:
            return BIG_NUM
        
        # otherwise return constant
        return 0.0
        

def main():
    # load and reduce image sizes
    reduce_factor = 3
    image_1 = Image.open("data/photo_1.jpg").reduce(reduce_factor)
    image_2 = Image.open("data/photo_2.jpg").reduce(reduce_factor)

    # let lambda be scale factor
    max_diffrence = np.max(np.linalg.norm(np.array(image_1) - np.array(image_2), axis=2))
    print(f"Max diffrence = {max_diffrence}")
    lambd_scale = (1 / max_diffrence) * 2.5
    lambd_array = [lambd_scale]

    for lambd in lambd_array:
        print(f"Cur lambda = {lambd}")

        graph_cut = MyGhraphCut(
            image_1,
            image_2,
            lambd
        )

        # obtain collage matrix and collage
        print("Final energy:", graph_cut.compose_collage())

        # make result's folder
        cur_results_path = f"experiments/experiment_exp/results_lambda_another3"
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
