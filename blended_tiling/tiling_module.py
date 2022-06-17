import math
from typing import cast, List, Optional, Tuple, Union

import torch


class TilingModule(torch.nn.Module):
    """
    This module can split NCHW tensors into overlapping tiles, and can blend them back
    together.
    """

    __constants__ = ["_tile_size", "_tile_overlap"]

    def __init__(
        self,
        tile_size: Union[int, List[int], Tuple[int, int]] = [224, 224],
        tile_overlap: Union[float, List[float], Tuple[float, float]] = [0.25, 0.25],
        base_size: Union[int, List[int], Tuple[int, int]] = [512, 512],
    ) -> None:
        """
        Args:

            tile_size (int or list of int): The size of tiles to use. A single
                integer to use for both the height and width dimensions, or a list
                of dimensions with a shape of: [height, width].
                Default: [224, 224]
            tile_overlap (int or list of int): The amount of overlap to use when
                creating tiles. A single integer to use for both the height and width
                dimensions, or a list of dimensions with a shape of: [height, width].
                Default = [0.25, 0.25]
            base_size (int or list of int): The size of the NCHW tensor being split
                into tiles. A single integer to use for both the height and width
                dimensions, or a list of dimensions with a shape of: [height, width].
                Default: [512, 512]
        """
        super().__init__()
        self._tile_size = cast(Tuple[int, int], tuple(self._prep_values(tile_size)))
        self._tile_overlap = cast(
            Tuple[float, float], tuple(self._prep_values(tile_overlap))
        )
        assert all([0.0 <= o <= 0.5 for o in self._tile_overlap])
        base_size = cast(List[int], list(self._prep_values(base_size)))
        assert self._tile_size[0] <= base_size[0] and self._tile_size[1] <= base_size[1]
        tiles, self._coords, self._overlap = self._get_tiles_and_coords(
            torch.ones([1, 1] + base_size)
        )
        self._num_tiles = tiles.shape[0]

    def _prep_values(
        self,
        size: Union[
            int, float, List[int], Tuple[int, int], List[float], Tuple[float, float]
        ],
    ) -> Union[List[int], List[float], Tuple[int, int], Tuple[float, float]]:
        """
        Format input variables into lists or tuples of 2 values.

        Args:
            size (int, float, tuple of int, tuple of float, list of int, or list of
                float): Values to ensure are part of a list or tuple with a length of
                2.

        Returns:
            size (tuple of int, list of int, tuple of float, or list of float): List
                or tuples of values with a length of 2.

        """
        size = [size] * 2 if not isinstance(size, (list, tuple)) else size
        assert len(size) == 2
        return size

    @torch.jit.export
    def num_tiles(self) -> int:
        """
        Returns:
            num_tiles (int): The number of tiles that the full image shape is divided
                into based on specified parameters.
        """
        return self._num_tiles

    @torch.jit.export
    def tiling_pattern(self) -> List[int]:
        """
        Returns:
            pattern (list of int): The number of tiles per column and number of tiles
                per row, in the format of: [n_tiles_per_column, n_tiles_per_row].
        """
        return [len(self._coords[0]), len(self._coords[1])]

    @torch.jit.export
    def get_tile_masks(
        self,
        channels: int = 3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        """
        Args:

            channels (int, optional): The number of channels to use for the masks.
                Default: 3
            device (torch.device, optional): The desired device to create the masks on.
                Default: torch.device("cpu")
            dtype (torch.dtype, optional): The desired dtype to create the masks with.
                Default: torch.float

        Returns:
            masks (torch.Tensor): A set of tile masks stacked across the batch
                dimension.
        """
        shape = [self._num_tiles, channels] + list(self._tile_size)
        return self._create_tile_masks(shape, device, dtype)

    def _calc_tile_coords(
        self, d: int, tile_dim: int, overlap: float = 0.0
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Args:

            d (int): The total size of the base image dimension.
            tile_dim (int): The size of the tile dimension.
            overlap (float, optional): The percentage of overlap to use.
                Default: 0.0

        Returns:
            coords (list of int): The starting and ending coords for each tile.
            overlap (list of lust of int): The amount of overlap to use for each
                tile edge.
        """
        c, tile_start, coords, overlaps = (
            1,
            0,
            [0],
            torch.jit.annotate(List[List[int]], []),
        )
        while tile_start + tile_dim < d:
            tile_start = int(tile_dim * (1 - overlap)) * c
            coords.append(
                d - tile_dim
            ) if tile_start + tile_dim >= d else coords.append(tile_start)

            lin_size = int((math.ceil(tile_dim * overlap)))
            lin_size = lin_size if lin_size > 1 else 0
            # Special overlap for right and bottom edge tiles
            if tile_start + tile_dim >= d:
                i = len(coords) - 1
                zeros_size = ((coords[i - 1] + tile_dim) - lin_size) - coords[i]
                # Only use lin_mask for a row / column of 2 with equal overlap
                if tile_dim > d / 2 and tile_dim != d:
                    lin_size = lin_size + zeros_size
                    zeros_size = 0
                ovlp = [zeros_size, lin_size]
            else:
                ovlp = [0, lin_size]
            if len(overlaps) == 0:
                overlaps.append(ovlp)
            overlaps.append(ovlp)

            c += 1
        return coords, overlaps

    def _get_tiles_and_coords(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[int]], List[List[List[int]]]]:
        """
        Split an NCHW tensor into tiles using calculated coordinates and overlap
        values.

        Args:

            tensor (torch.Tensor): An NCHW tensor to split into tiles.

        Returns:
            tiles (torch.Tensor): A set of NCHW tiles created from the input tensor,
                stacked across the batch dimension.
            coords (list of list of int): Sets of x and y coordinates corresponding
                to each tile.
            overlaps (list of list of list of int): Sets of x and y overlaps
                corresponding to each tile / set of coordinates.
        """
        assert tensor.dim() == 4 and tensor.shape[0] == 1

        y_coords, y_overlaps = self._calc_tile_coords(
            tensor.shape[2], self._tile_size[0], self._tile_overlap[0]
        )
        x_coords, x_overlaps = self._calc_tile_coords(
            tensor.shape[3], self._tile_size[1], self._tile_overlap[1]
        )
        tile_coords = torch.jit.annotate(List[Tuple[int, int, int, int]], [])
        [
            [
                tile_coords.append(
                    (y, y + self._tile_size[0], x, x + self._tile_size[1])
                )
                for x in x_coords
            ]
            for y in y_coords
        ]
        tiles = torch.cat([tensor[..., c[0] : c[1], c[2] : c[3]] for c in tile_coords])
        return tiles, [y_coords, x_coords], [y_overlaps, x_overlaps]

    def _create_mask_part(
        self,
        shape: List[int],
        overlap: List[int],
        device: torch.device,
        dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        """
        Create part of a mask for a specified side.

        Args:

            shape (list of int): The shape of the tile mask being created.
            overlap (list of int): The amount of overlap being used.
            device (torch.device, optional): The desired device to create the mask on.
                Default: torch.device("cpu")
            dtype (torch.dtype, optional): The desired dtype to create the masks with.
                Default: torch.float

        Returns:
            mask_part (torch.Tensor): The mask for the specified side.
        """
        zeros_size, lin_size = overlap[0:2]
        ones_size = shape[3] - (zeros_size + lin_size)
        sizes = (zeros_size, lin_size, ones_size)
        mask_parts = [
            torch.zeros(sizes[0], device=device, dtype=dtype),
            torch.linspace(0, 1, sizes[1], device=device, dtype=dtype),
            torch.ones(sizes[2], device=device, dtype=dtype),
        ]
        return (
            torch.cat(mask_parts, 0)
            .repeat(shape[2], 1)
            .repeat(shape[1], 1, 1)
            .unsqueeze(0)
        )

    def _build_mask(
        self,
        position: int,
        grid_dim: int,
        rot_list: List[int],
        shape: List[int],
        ovlp: List[int],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        """
        Build individual tile masks.

        Args:

            position (int): The dimension being masked.
            grid_dim (int): The number of tiles being used for the specified position.
            rot_list (list of int): The amount to rotate the mask so that it masks the
                appropriate side.
            shape (list of int): The full size of the tile being masked.
            overlap (list of int): A list of overlap values to use.
            device (torch.device, optional): The desired device to create the mask on.
                Default: torch.device("cpu")
            dtype (torch.dtype, optional): The desired dtype to create the masks with.
                Default: torch.float
        """
        # Mask right / bottom side
        if position == 0:
            mask = self._create_mask_part(shape, ovlp, device, dtype).rot90(
                rot_list[0], [2, 3]
            )
        # Mask left & right or top & bottom sides
        elif position > 0 and position < grid_dim - 1:
            mask = self._create_mask_part(shape, ovlp, device, dtype).rot90(
                rot_list[0], [2, 3]
            )
            mask = mask * self._create_mask_part(shape, ovlp, device, dtype).rot90(
                rot_list[1], [2, 3]
            )
        # Mask left / top side
        else:
            mask = self._create_mask_part(shape, ovlp, device, dtype).rot90(
                rot_list[1], [2, 3]
            )
        return mask

    def _create_tile_masks(
        self,
        tile_shape: List[int],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        """
        Args:

            tile_shape (list of int): The shape of the tiles being used, in the
                format of: [1, channels, height, width].
            device (torch.device, optional): The desired device to create the mask on.
                Default: torch.device("cpu")
            dtype (torch.dtype, optional): The desired dtype to create the masks with.
                Default: torch.float

        Returns:
            masks (torch.Tensor): A stack of tile masks.
        """
        tile_coords, tile_overlaps = self._coords, self._overlap
        assert len(tile_coords) == 2 and len(tile_shape) == 4
        masks = []
        for column, y in enumerate(tile_coords[0]):
            for row, x in enumerate(tile_coords[1]):
                tile_mask = torch.ones(
                    [1] + list(tile_shape[1:]), device=device, dtype=dtype
                )

                # Vertical masking along H dim (Top & Bottom)
                if tile_coords[0] != [0]:
                    tile_mask_h = self._build_mask(
                        position=column,
                        grid_dim=len(tile_coords[0]),
                        rot_list=[1, 3],
                        shape=tile_shape[:2] + tile_shape[2:][::-1],
                        ovlp=tile_overlaps[0][column],
                        device=device,
                        dtype=dtype,
                    )
                    tile_mask = tile_mask * tile_mask_h

                # Horizontal masking along W dim (Left & Right)
                if tile_coords[1] != [0]:
                    tile_mask_w = self._build_mask(
                        position=row,
                        grid_dim=len(tile_coords[1]),
                        rot_list=[2, 0],
                        shape=tile_shape,
                        ovlp=tile_overlaps[1][row],
                        device=device,
                        dtype=dtype,
                    )
                    tile_mask = tile_mask * tile_mask_w

                masks.append(tile_mask)
        return torch.cat(masks)

    def _color_borders(
        self,
        x: torch.Tensor,
        border: int = 1,
        colors: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Color tile borders for explainability and debugging.

        Args:

            x (torch.Tensor) A tile to add a border to.
            border (int, optional): The size of border to use.
                Default: 1
            colors (list of float, optional): A set of floats for each image channel
                to use for the border color.
        """
        colors = colors if colors is not None else [1.0, 0.0, 0.0]
        assert x.dim() == 4 and x.shape[1] == len(colors) and border > 0
        x = torch.nn.functional.pad(x, [-border] * 4)
        x_channels = [x[:, c : c + 1].clone() for c in range(x.shape[1])]
        new_channels, pad = [], [border] * 4
        for x_channel, color_c in zip(x_channels, colors):
            new_channels.append(
                torch.nn.functional.pad(x_channel, pad, mode="constant", value=color_c)
            )
        return torch.cat(new_channels, dim=1)

    def rebuild(
        self,
        tiles: torch.Tensor,
        border: Optional[int] = None,
        colors: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Args:

            tiles (torch.Tensor): A set of tiles that may or not be masked, stacked
                across the batch dimension.
            border (int, optional): Optionally add a red border of a specified size to
                the edges of tiles in the full image for debugging and explainability.
                Set to None for no border.
                Default: None
            colors (list of float, optional): A set of floats to use for the border
                color, if using borders.
                Default: None

        Returns:
            full_image (torch.Tensor): The full image made up of tiles blended
                together.
        """
        assert tiles.dim() == 4
        tile_coords = self._coords

        tile_size = list(tiles.shape[2:])
        h = tile_coords[0][len(tile_coords[0]) - 1] + tile_size[0]
        w = tile_coords[1][len(tile_coords[1]) - 1] + tile_size[1]
        base_tensor = torch.zeros(
            1, tiles.shape[1], h, w, device=tiles.device, dtype=tiles.dtype
        )

        i = 0
        for column, y in enumerate(tile_coords[0]):
            for row, x in enumerate(tile_coords[1]):
                tile = tiles[i : i + 1].clone()
                base_tensor[..., y : y + tile_size[0], x : x + tile_size[1]] = (
                    base_tensor[..., y : y + tile_size[0], x : x + tile_size[1]] + tile
                )
                i += 1
                if self._num_tiles > tiles.shape[0] and i == tiles.shape[0]:
                    return base_tensor

        if border is not None:
            i = 0
            for column, y in enumerate(tile_coords[0]):
                for row, x in enumerate(tile_coords[1]):
                    tile = tiles[i : i + 1].clone()
                    base_tensor[
                        ..., y : y + tile_size[0], x : x + tile_size[1]
                    ] = self._color_borders(
                        x=base_tensor[..., y : y + tile_size[0], x : x + tile_size[1]],
                        border=border,
                        colors=colors,
                    )
                    i += 1
        return base_tensor

    def rebuild_with_masks(
        self,
        x: torch.Tensor,
        border: Optional[int] = None,
        colors: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): A set of tiles to use for creating the full image.
            border (int, optional): Optionally add a red border of a specified size to
                the edges of tiles in the full image for debugging and explainability.
                Set to None for no border.
                Default: None
            colors (list of float, optional): A set of floats to use for the border
                color, if using borders.
                Default: None

        Returns:
            full_image (torch.Tensor): The full image made up of tiles blended
                together.
        """
        assert x.dim() == 4
        masks = self._create_tile_masks(list(x.shape), x.device, x.dtype)[: x.shape[0]]
        return self.rebuild(x * masks, border, colors)

    @torch.jit.export
    def split_into_tiles(self, x: torch.Tensor):
        """
        Split an NCHW image tensor into tiles, and set the stored coordinates and
        overlap values to match the new image.

        Args:

            x (torch.Tensor): An NCHW tensor to split into tiles.

        Returns:
            tiles (torch.Tensor): A set of tiles created from the input image.
        """
        tiles, self._coords, self._overlap = self._get_tiles_and_coords(x)
        self._num_tiles = tiles.shape[0]
        return tiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine a set of NCHW tiles stacked across the batch dimension into the full
        image, so that overlapping areas are blended together. Then split the tiles
        apart again.

        Args:

            x (torch.Tensor): A set of tiles to blend the overlapping regions
                together of.

        Returns:
            x (torch.Tensor): A set of tiles with overlapping regions blended
                together.
        """
        full_tensor = self.rebuild_with_masks(x)
        return self._get_tiles_and_coords(full_tensor)[0]


__all__ = ["TilingModule"]
