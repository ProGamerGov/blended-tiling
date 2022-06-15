#!/usr/bin/env python3
import random
import unittest

import numpy as np
import torch
from blended_tiling import TilingModule


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def assertTensorAlmostEqual(
    self, actual: torch.Tensor, expected: torch.Tensor, delta: float = 0.0001
) -> None:
    """
    Args:

        self (): A unittest instance.
        actual (torch.Tensor): A tensor to compare with expected.
        expected (torch.Tensor): A tensor to compare with actual.
        delta (float, optional): The allowed difference between actual and expected.
            Default: 0.0001
    """
    assert actual.shape == expected.shape
    assert actual.device == expected.device
    self.assertAlmostEqual(
        torch.sum(torch.abs(actual - expected)).item(), 0.0, delta=delta
    )


class TestTilingModule(BaseTest):
    def test_init_function(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )
        self.assertEqual(tiling_module._tile_size, tuple(tile_size))
        self.assertEqual(tiling_module._tile_overlap, tuple(tile_overlap))
        self.assertEqual(tiling_module._coords, [[0, 168, 288], [0, 168, 288]])
        self.assertEqual(
            tiling_module._overlap,
            [[[0, 56], [0, 56], [48, 56]], [[0, 56], [0, 56], [48, 56]]],
        )
        self.assertEqual(tiling_module.num_tiles(), 9)
        self.assertEqual(tiling_module.tiling_pattern(), [3, 3])

    def test_init_function_jit_module(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )
        tiling_module = torch.jit.script(tiling_module)
        self.assertEqual(tiling_module._tile_size, tuple(tile_size))
        self.assertEqual(tiling_module._tile_overlap, tuple(tile_overlap))
        self.assertEqual(tiling_module._coords, [[0, 168, 288], [0, 168, 288]])
        self.assertEqual(
            tiling_module._overlap,
            [[[0, 56], [0, 56], [48, 56]], [[0, 56], [0, 56], [48, 56]]],
        )
        self.assertEqual(tiling_module.num_tiles(), 9)
        self.assertEqual(tiling_module.tiling_pattern(), [3, 3])

    def test_forward_basic_square(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_basic_rectangle_h(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 512]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([num_tiles, 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.0)

    def test_forward_basic_rectangle_w(self) -> None:
        full_size = [512, 512]
        tile_size = [512, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([num_tiles, 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.0)

    def test_forward_basic_square_jit_module(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        tiling_module = torch.jit.script(tiling_module)
        num_tiles = tiling_module.num_tiles()
        x = torch.ones([num_tiles, 3] + tile_size, dtype=torch.float)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_basic_square_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping basic forward CUDA test due to not supporting CUDA."
            )
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )
        num_tiles = tiling_module.num_tiles()
        x = torch.ones([num_tiles, 3] + tile_size).cuda()

        output = tiling_module(x)
        self.assertTrue(output.is_cuda)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.0)

    def test_forward_basic_square_manual(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )
        x = torch.ones([1, 3] + full_size, dtype=torch.float)

        tiles = tiling_module.split_into_tiles(x)
        masks = tiling_module.get_tile_masks()
        output = tiling_module.rebuild(tiles * masks)

        self.assertEqual(list(masks.shape), list(tiles.shape))
        assertTensorAlmostEqual(self, tiles, torch.ones_like(tiles), delta=0.0)
        assertTensorAlmostEqual(self, output, torch.ones_like(output), delta=0.0008)

    def test_forward_basic_square_manual_jit_module(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )
        tiling_module = torch.jit.script(tiling_module)
        x = torch.ones([1, 3] + full_size, dtype=torch.float)

        tiles = tiling_module.split_into_tiles(x)
        masks = tiling_module.get_tile_masks()
        output = tiling_module.rebuild(tiles * masks)

        self.assertEqual(list(masks.shape), list(tiles.shape))
        assertTensorAlmostEqual(self, tiles, torch.ones_like(tiles), delta=0.0)
        assertTensorAlmostEqual(self, output, torch.ones_like(output), delta=0.0008)

    def test_forward_lin_mask_0(self) -> None:
        full_size = [64 * 2, 64 * 2]
        tile_size = [9 * 2, 9 * 2]

        tile_overlap = [0.10, 0.10]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_2_row_2_column(self) -> None:
        full_size = [64, 64]
        tile_size = [38, 38]

        tile_overlap = [0.24, 0.24]

        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_2_row_2_column_times_2(self) -> None:
        full_size = [64 * 2, 64 * 2]
        tile_size = [38 * 2, 38 * 2]

        tile_overlap = [0.24, 0.24]

        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_basic_square_min_overlap(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.0, 0.0]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_basic_square_max_overlap(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.5, 0.5]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        assertTensorAlmostEqual(self, x, output, delta=0.04)

    def test_forward_basic_square_autograd(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        num_tiles = tiling_module.num_tiles()
        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size, requires_grad=True)
        output = tiling_module(x)
        self.assertEqual(list(output.shape), [num_tiles, 3] + tile_size)
        self.assertTrue(output.requires_grad)
        assertTensorAlmostEqual(self, x, output, delta=0.003)

    def test_forward_basic_square_rebuild_with_masks(self) -> None:
        full_size = [512, 512]
        tile_size = [224, 224]
        tile_overlap = [0.25, 0.25]
        tiling_module = TilingModule(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            base_size=full_size,
        )

        x = torch.ones([tiling_module.num_tiles(), 3] + tile_size)
        output = tiling_module.rebuild_with_masks(x)
        self.assertEqual(list(output.shape), [1, 3] + full_size)
        assertTensorAlmostEqual(self, torch.ones_like(output), output, delta=0.003)
