import numpy as np
import random


class TSPGeometry:
    """
    Helper class takes nodes and applies reflections,
    rotations, and scaling to the nodes, while keeping everything within the [0,1]^2 unit square.

    In all methods, `nodes` is expected to be a numpy array of shape (<num_nodes>, 2)

    Important: Some operations are in-place, so make sure to copy the nodes before
    """
    @staticmethod
    def linear_scale(nodes: np.array, scale_factor: float) -> np.array:
        """
        Performs linear scaling of the nodes.
        Parameters
        ----------
            scale_factor: [float] Maps (x, y) to (scale_factor * x, scale_factor * y)
        """
        if not (-1e-11 <= scale_factor <= 1 + 1e-11):
            print(f"WARNING: TSPGeometry performs linear scaling with a scale_factor of {scale_factor}. Cannot"
                  f" guarantee that all nodes lie in unit square again.")
        return nodes * scale_factor

    @staticmethod
    def swap_x_y(nodes: np.array) -> np.array:
        """
        Swap x/y coordinate
        """
        x = nodes[:, 0].copy()
        y = nodes[:, 1].copy()
        nodes[:, 1] = x
        nodes[:, 0] = y
        return nodes

    @staticmethod
    def reflection(nodes: np.array, direction: int) -> np.array:
        """
        Performs a reflection of the board within the unit square (i.e. translation of (0.5, 0.5) to origin, reflecting
        and back)
        Parameters
        ----------
        direction: [int] 0 for horizontal reflection, 1 for vertical reflection
        """
        if direction not in [0, 1]:
            print("WARNING: TSPGeometry reflection only takes 0 and 1 as direction. Not executing transformation.")
            return nodes

        nodes[:, direction] = -1 * nodes[:, direction] + 1
        return nodes

    @staticmethod
    def rotate(nodes: np.array, angle: float) -> np.array:
        """
        Performs rotation around center (0.5, 0.5) of unit square by `angle` given in radians.
        As the rotation does not guarantee that all nodes lie again in the unit square, afterwards a
        linear scaling is performed if necessary.

        Parameters
        ----------
        angle: [float] Rotation angle in radians
        """
        max_coordinate = -1  # keeps track of maximum absolute coordinate to apply scaling if necessary
        cos = np.cos(angle)
        sin = np.sin(angle)

        nodes = TSPGeometry.translate(nodes, -0.5, -0.5)

        new_x = cos * nodes[:, 0] - sin * nodes[:, 1]
        new_y = sin * nodes[:, 0] + cos * nodes[:, 1]
        nodes[:, 0] = new_x
        nodes[:, 1] = new_y

        # We have translated the board to the origin and rotated. Might be that now the absolute value
        # of a coordinate is greater than 0.5, which means that if we translate back we will no longer be in
        # unit square
        if nodes.shape[0] != 0:
            max_coordinate = max(max_coordinate, np.max(np.abs(nodes)).item())

        if max_coordinate > 0.5:
            nodes = TSPGeometry.linear_scale(nodes, scale_factor=0.5 / max_coordinate)

        return TSPGeometry.translate(nodes, 0.5, 0.5)

    @staticmethod
    def translate(nodes: np.array, x: float, y: float) -> np.array:
        """
        Adds vector (x, y) to each node.
        """
        return nodes + np.array([x, y])[None, :]

    @staticmethod
    def random_state_augmentation(nodes: np.array, do_linear_scale: bool = False) -> np.array:
        """
        Performs a random reflection, scaling and rotation to the board.
        """
        augments = random.choice(
            [None, ["swap"], [0, 1], [0], [1], [0, 1, "swap"], [0, "swap"], [1, "swap"]]
        )

        if augments is not None:
            for action in augments:
                if action == "swap":
                    nodes = TSPGeometry.swap_x_y(nodes)
                else:
                    nodes = TSPGeometry.reflection(nodes, int(action))

        # Apply random rotation
        angle = np.random.random() * 2 * np.pi
        nodes = TSPGeometry.rotate(nodes, angle)

        if do_linear_scale:
            nodes = nodes * np.random.uniform(low=0.5, high=1.0)

        return nodes