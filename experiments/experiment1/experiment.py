import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# add src/ folder
from sys import path
path.append("/home/cyrill/Sem8/Byas/prac1/")
from src.exp_potential import ExpGraphCut
from src.poly_potential import PolyGraphCut

class MyGhraphCut(PolyGraphCut):
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

        # image 0 pixels to include for sure
        if (200 <= i <= 250) and (150 <= j <= 250) and value == 1:
            return np.inf
        if (290 <= i <= 310) and (410 <= j <= 430) and value == 1:
            return np.inf

        # image 1 pixels to include for sure
        if (325 <= i <= 350) and (400 <= j <= 600) and value == 0:
            return np.inf
        if (250 <= i <= 310) and (10 <= j <= 30) and value == 0:
            return np.inf
        
        # otherwise return constant
        return 0
        

def main():
    # load and reduce image sizes
    reduce_factor = 2
    image_1 = Image.open("data/photo_1.jpg").reduce(reduce_factor)
    image_2 = Image.open("data/photo_2.jpg").reduce(reduce_factor)

    graph_cut = MyGhraphCut(
        image_1,
        image_2,
        alpha=1.8
    )

    # obtain collage matrix and collage
    print("Final energy:", graph_cut.compose_collage())

    # save collage matrix
    fig, ax = plt.subplots()
    ax.imshow(graph_cut.collage_matrix, cmap="binary")
    # legend
    img_0_patch = mpatches.Patch(color='white', label='img_0')
    img_1_patch = mpatches.Patch(color='black', label='img_1')
    ax.legend(handles=[img_0_patch, img_1_patch])

    fig.savefig("experiments/experiment1/results/collage_matrix.png", format="png")

    # save collage
    Image.fromarray(graph_cut.collage).save("experiments/experiment1/results/collage.png", format="png")


if __name__ == "__main__":
    main()
