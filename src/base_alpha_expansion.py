import numpy as np
from PIL import Image
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp, preflow_push

from abc import ABC, abstractmethod
from copy import deepcopy

# debug
from time import time


class BaseAlphaExpansion(ABC):
    def __init__(self, 
                 images: list,  # list of PIL.Image
                 init_collage_matrix: np.ndarray = None
    ) -> None:
        self.images_ = []
        for image in images:
            self.images_.append(np.array(image, dtype=np.uint8))
        self.num_classes_ = len(images)

        # set initial collage matrix
        if init_collage_matrix is None:
            image_size = self.images_[0].shape[0:2]
            self._collage_matrix_ = np.random.randint(0, self.num_classes_, size=image_size, dtype=np.int32)
        else:
            self._collage_matrix_ = init_collage_matrix.copy()

        # all collage matrices during algorithm iterations
        self._collage_matrix_seq_ = [self._collage_matrix_.copy()]
        # final collage-photo
        self._collage_ = None

        # array of minimum energies during iterations
        self._energies_ = []

        # constructing graph
        self._graph_ = nx.DiGraph()
        self._init_graph()

    def _init_graph(self):
        """initializes graph for CutGraph procedure for reuse during iterations. Only capacities
            will be required to set
        """
        im_height, im_width = self.images_[0].shape[0:2]

        # debug
        print("Initing graph")

        # nodes
        self._graph_.add_nodes_from(
            [(i, j) for i in range(im_height) for j in range(im_width)]
        )
        # add source and target nodes
        self._graph_.add_node("s")
        self._graph_.add_node("t")

        # edges
        for i in range(im_height):
            for j in range(im_width):
                cur_node = (i, j)

                # add source and target edges
                self._graph_.add_edge("s", cur_node, capacity=-1.)
                self._graph_.add_edge(cur_node, "t", capacity=-1.)

                neighbours = self.get_neighbours(i, j)
                for neigb in neighbours:
                    # add outgoing from cur_node to neighbours edge
                    self._graph_.add_edge(cur_node, 
                                   neigb, 
                                   capacity=-1.
                    )

    @property
    def collage_matrix_seq(self) -> np.ndarray:
        return self._collage_matrix_seq_
    
    @property
    def collage(self) -> np.ndarray:
        return self._collage_
    
    @property
    def energies(self) -> list:
        return self._energies_
    
    @abstractmethod
    def uno_potential(self, i: int, j: int, value: int, cur_alpha: int) -> float:
        ...

    @abstractmethod
    def duel_potential(self, i: int, j: int, n: int, k: int, value_1: int, value_2: int, cur_alpha: int) -> float:
        ...

    @classmethod
    @abstractmethod
    def get_neighbours(cls, i: int, j: int) -> list:
        ...

    def alpha_expansion(self, max_iter: int = 10) -> None:
        """implemented iterations of alpha expansion algorithm
        """
        # reset all previous iteration results
        self._reset_results()

        for iter in range(max_iter):
            cur_alpha = iter % self.num_classes_

            # debug
            print(f"Iter {iter}; Current alpha = {cur_alpha}")

            # check whether some pixels have changed
            if self._alpha_exp_iter(cur_alpha):
                # debug
                print(f"Last energy = {self._energies_[-1]}")
                
                break

            # debug
            print(f"Last energy = {self._energies_[-1]}\n")

        # build collage
        im_height, im_width = self.images_[0].shape[0:2]
        self._collage_ = np.zeros(self.images_[0].shape, dtype=np.uint8)
        for i in range(im_height):
            for j in range(im_width):
                cur_class = self._collage_matrix_[i][j]
                self._collage_[i][j] = self.images_[cur_class][i][j]

        # debug
        print("Collage's built")

    
    def _reset_results(self):
        self._collage_matrix_seq_ = self._collage_matrix_seq_[0:1]
        self._collage_matrix_ = self._collage_matrix_seq_[0].copy()
        self._energies_ = []
        self._collage_ = None

    def _alpha_exp_iter(self, cur_alpha: int) -> bool:
        """solve GraphCut problem for current alpha

        Returns: whether no pixels were changed
        """
        # debug
        time_past = time()

        im_height, im_width = self.images_[0].shape[0:2]
        # set graph edges according to current alpha
        for i in range(im_height):
            for j in range(im_width):
                cur_node = (i, j)

                # debug
                temp_1 = self.uno_potential(
                    i, j, 1, cur_alpha
                )
                temp_2 = self.uno_potential(
                    i, j, 0, cur_alpha
                )
                # if temp_1 != 0 and temp_1 == temp_2:
                #     raise(f"({i}, {j}) has both uno pot = inf")
                
                # if temp_1 != 0:
                #     #print("temp_1")
                # if temp_2 != 0:
                #     print("temp_2")

                # add source and target edges
                self._graph_["s"][cur_node]["capacity"] = self.uno_potential(
                    i, j, 1, cur_alpha
                )
                self._graph_[cur_node]["t"]["capacity"] = self.uno_potential(
                    i, j, 0, cur_alpha
                )

                neighbours = self.get_neighbours(i, j)
                for neigb in neighbours:
                    # add outgoing from cur_node to neighbours edge
                    self._graph_[cur_node][neigb]["capacity"] = self.duel_potential(
                        i, j, neigb[0], neigb[1], 0, 1, cur_alpha
                    )

        # debug
        print(f"Setting edges: Okey; Time = {time() - time_past}")
        time_past = time()

        # debug
        # max_flow, _ = nx.maximum_flow(self._graph_, "s", "t", flow_func=preflow_push)
        # print(f"Max flow: Okey; Time = {time() - time_past}")
        # print(f"Max flow = {max_flow}")
        # time_past = time()

        # solve minimum cut problem
        min_energy, (left_partition, right_partition) = nx.minimum_cut(self._graph_, "s", "t", flow_func=preflow_push)

        # debug
        print(f"Mincut: Okey; Time = {time() - time_past}")
        print(f"Pixels changed: {(len(right_partition) - 1) / (im_height * im_width)}")
        time_past = time()

        # update collage matrix with new alpha-class pixels
        for node in right_partition:
            if node != "t":
                i, j = node
                self._collage_matrix_[i][j] = cur_alpha

        # save new collage matrix and energy
        self._collage_matrix_seq_.append(self._collage_matrix_.copy())
        self._energies_.append(min_energy)

        # debug
        print(f"Collage matrix update: Okey; Time = {time() - time_past}")
        time_past = time()

        return len(right_partition) <= 1
