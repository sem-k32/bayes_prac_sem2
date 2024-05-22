import numpy as np
from PIL import Image
import networkx as nx

from abc import ABC, abstractmethod

# debug
from time import time


class BaseGraphCut(ABC):
    def __init__(self, image_1: Image, image_2: Image) -> None:
        self.images_ = [np.array(image_1), np.array(image_2)]

        self._collage_matrix_ = None
        # collage-photo
        self._collage_ = None

    @property
    def collage_matrix(self) -> np.ndarray:
        return self._collage_matrix_
    
    @property
    def collage(self) -> np.ndarray:
        return self._collage_
    
    @abstractmethod
    def uno_potential(self, i: int, j: int, value: int) -> float:
        ...

    @abstractmethod
    def duel_potential(self, i: int, j: int, n: int, k: int, value_1: int, value_2: int) -> float:
        ...

    @classmethod
    @abstractmethod
    def get_neighbours(cls, i: int, j: int) -> list:
        ...

    def compose_collage(self) -> float:
        """solve GraphCut problem and saves collage and collage matrix

        Returns:
            float: value of minimum energy
        """
        # debug
        time_past = time()

        im_height, im_width = self.images_[0].shape[0:2]

        # constructing graph
        graph = nx.DiGraph()

        # nodes
        graph.add_nodes_from(
            [(i, j) for i in range(im_height) for j in range(im_width)]
        )
        # add source and target nodes
        graph.add_node("s")
        graph.add_node("t")

        # debug
        print(f"Nodes: Okey; Time = {time() - time_past}")
        time_past = time()

        # edges
        for i in range(im_height):
            for j in range(im_width):
                cur_node = (i, j)

                # add source and target edges
                graph.add_edge("s", cur_node, capacity=self.uno_potential(i, j, 1))
                graph.add_edge(cur_node, "t", capacity=self.uno_potential(i, j, 0))

                neighbours = self.get_neighbours(i, j)
                for neigb in neighbours:
                    # add outgoing from cur_node to neighbours edge
                    graph.add_edge(cur_node, 
                                   neigb, 
                                   capacity=self.duel_potential(i, j, neigb[0], neigb[1], 0, 1)
                    )

        # debug
        print(f"Edges: Okey; Time = {time() - time_past}")
        time_past = time()

        # solve minimum cut problem
        min_energy, (left_partition, right_partition) = nx.minimum_cut(graph, "s", "t")

        # debug
        print(f"Mincut: Okey; Time = {time() - time_past}")
        time_past = time()

        # construct collage and collage matrix
        self._collage_matrix_ = np.zeros((im_height, im_width), dtype=bool)
        self._collage_ = np.empty_like(self.images_[0])

        for node in left_partition:
            if node != "s":
                i, j = node
                self._collage_[i][j] = self.images_[0][i][j]

        for node in right_partition:
            if node != "t":
                i, j = node
                self._collage_[i][j] = self.images_[1][i][j]
                self._collage_matrix_[i][j] = 1

        # debug
        print(f"Collage: Okey; Time = {time() - time_past}")
        time_past = time()

        return min_energy


