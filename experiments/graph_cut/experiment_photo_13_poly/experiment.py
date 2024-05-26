import numpy as np
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# add src/ folder
from sys import path
path.append(".")
from src.graph_cut.poly_potential import PolyGraphCut

import pathlib


class MyGhraphCut(PolyGraphCut):
    """final graph cut class with uno potential
    """
    def __init__(self, image: Image, lambd: float, beta: float) -> None:
        super().__init__(image, lambd, beta)

        # learn "water pixel"
        self._water_pixel_ = np.mean(self.image_[200:250, 200:300], axis=(0, 1))
        # find scale factor for uno potential
        self._uno_potent_scale_ = np.sum(np.maximum(
            np.abs(self._water_pixel_ - np.array([0, 0, 0])),
            np.abs(self._water_pixel_ - np.array([255, 255, 255]))
        ))

    def uno_potential(self, i: int, j: int, value: int) -> float:
        # compute metric, that current pixel is water
        cur_prob = np.linalg.norm(self.image_[i][j] - self._water_pixel_, ord=1)
        cur_prob /= self._uno_potent_scale_

        # debug
        if 1 - cur_prob < 0:
            raise ValueError("uno potential is incorrect")
        
        # energy for water
        if value == 0:
            return cur_prob
        else:
            return 1 - cur_prob
        

def main():
    # load and reduce image sizes
    reduce_factor = 5
    image = Image.open("data/archive/Image/13.jpg").reduce(reduce_factor)

    lambd = 1.5
    # choose scale factor for poly potential
    beta = -1 / np.linalg.norm(np.array([255, 255, 255]), ord=1)

    print(f"Cur lambda = {lambd}")

    graph_cut = MyGhraphCut(
        image,
        lambd,
        beta
    )

    # obtain segmentation
    print("Final energy:", graph_cut.segment_image())

    # make result's folder
    cur_results_path = f"experiments/graph_cut/experiment_photo_13_poly/results_init"
    pathlib.Path(cur_results_path).mkdir(exist_ok=True)

    # save segmentation mask
    fig, ax = plt.subplots()
    ax.imshow(graph_cut.seg_matrix, cmap="binary")
    # legend
    img_0_patch = mpatches.Patch(color='white', label='water')
    img_1_patch = mpatches.Patch(color='black', label='rest')
    ax.legend(handles=[img_0_patch, img_1_patch])

    fig.savefig(f"{cur_results_path}/seg_matrix.png", format="png")

    # save segmented photo
    fig, ax = plt.subplots()

    ax.imshow(image)

    ax.imshow(~graph_cut.seg_matrix, cmap="RdYlBu", alpha=0.4)
    # legend
    img_0_patch = mpatches.Patch(color='blue', label='water')
    img_1_patch = mpatches.Patch(color='red', label='rest')
    ax.legend(handles=[img_0_patch, img_1_patch])

    fig.savefig(f"{cur_results_path}/seg_image.png", format="png")

    # compute deviation from handmade segmentation
    real_seg_mask = np.array(Image.open("data/archive/Mask/13.png").reduce(reduce_factor), dtype=bool)
    num_pixels = real_seg_mask.shape[0] * real_seg_mask.shape[1]
    accuracy = np.sum(real_seg_mask == ~graph_cut.seg_matrix) / num_pixels
    print(f"Segmentation accuracy = {accuracy}")
    with open(f"{cur_results_path}/accuracy.txt", "w") as f:
        f.write(f"Accuracy = {accuracy}")


if __name__ == "__main__":
    main()
