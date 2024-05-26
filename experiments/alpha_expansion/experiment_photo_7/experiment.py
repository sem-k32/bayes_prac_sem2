import numpy as np
from PIL import Image
from PIL import ImageOps
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# add src/ folder
from sys import path
path.append(".")
from src.alpha_expansion.norm_potential import NormAlphaExtension

import pathlib


class MyAlphaExpansion(NormAlphaExtension):
    def __init__(self, image: Image, num_classes: int, init_seg_matrix: np.ndarray = None, lambd: float = 1) -> None:
        super().__init__(image, num_classes, init_seg_matrix, lambd)

        # build init seg matrix 
        self._seg_matrix_ = self.build_init_seg_matr()
        self._seg_matrix_seq_ = [self._seg_matrix_.copy()]

    def uno_potential(self, i: int, j: int, value: int, cur_alpha: int) -> float:
        # compute metric, that current pixel is water
        water_prob = np.linalg.norm(self.image_[i][j] - self._water_pixel_, ord=1)
        water_prob /= self._water_scale_
        # compute metric, that current pixel is green
        green_prob = np.linalg.norm(self.image_[i][j] - self._green_pixel_, ord=1)
        green_prob /= self._green_scale_
        # compute metric, that current pixel is rest
        rest_prob = max(0., 1 - water_prob - green_prob)

        # we do not care about summing this numbers into 1

        cur_label = self._seg_matrix_[i][j] if value == 0 else cur_alpha
        if cur_label == 0:
            return water_prob
        elif cur_label == 1:
            return green_prob
        else:
            return rest_prob

        
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
    
    def build_init_seg_matr(self) -> np.ndarray:
        image_size = self.image_.shape[0:2]
        # simple init with class 1
        seg_matrix_init = np.ones(image_size, dtype=np.int32)

        # learn "water pixel"
        self._water_pixel_ = np.mean(self.image_[120:140, 480:490], axis=(0, 1))
        # find scale factor for uno potential
        self._water_scale_ = np.sum(np.maximum(
            np.abs(self._water_pixel_ - np.array([0, 0, 0])),
            np.abs(self._water_pixel_ - np.array([255, 255, 255]))
        ))

        # learn "green pixel"
        self._green_pixel_ = np.mean(self.image_[50:80, 180:190], axis=(0, 1))
        # find scale factor for uno potential
        self._green_scale_ = np.sum(np.maximum(
            np.abs(self._green_pixel_ - np.array([0, 0, 0])),
            np.abs(self._green_pixel_ - np.array([255, 255, 255]))
        ))

        return seg_matrix_init

        
def main():
     # load and reduce image sizes
    reduce_factor = 4
    image = Image.open("data/archive/Image/7.jpg").reduce(reduce_factor)

    # choose scale factor for poly potential
    lambd = 1 / np.linalg.norm(np.array([255, 255, 255]), ord=1)

    print(f"Cur lambda = {lambd}")

    segmenting_class = MyAlphaExpansion(
        image,
        num_classes=3,
        lambd=lambd
    )

    segmenting_class.alpha_expansion(max_iter=6)
    
    #print(f"Energies: {collage_class.energies}")

    # make result's folder
    cur_results_path = f"experiments/alpha_expansion/experiment_photo_7/results_init"
    pathlib.Path(cur_results_path).mkdir(exist_ok=True)

    # save collage matrices
    for iter, seg_matr in enumerate(segmenting_class.seg_matrix_seq):
        fig, ax = plt.subplots()
        ax.imshow(seg_matr)
        fig.savefig(f"{cur_results_path}/seg_matr_{iter}.png", format="png")

    # save segmented photo
    fig, ax = plt.subplots()

    ax.imshow(image)

    ax.imshow(segmenting_class.seg_matrix_seq[-1], 
              cmap = ListedColormap(['blue', 'green', 'red']), 
              vmin=0, vmax=3,
              alpha=0.4
    )
    # legend
    img_0_patch = mpatches.Patch(color='blue', label='water')
    img_1_patch = mpatches.Patch(color='green', label='greens')
    img_2_patch = mpatches.Patch(color='red', label='rest')
    ax.legend(handles=[img_0_patch, img_1_patch, img_2_patch])

    fig.savefig(f"{cur_results_path}/seg_image.png", format="png")


if __name__ == "__main__":
    main()
