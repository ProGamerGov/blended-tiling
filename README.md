# blended-tiling

[![DOI](https://zenodo.org/badge/503953108.svg)](https://zenodo.org/badge/latestdoi/503953108)

This module adds support for splitting NCHW tensor inputs like images & activations into overlapping tiles of equal size, and then blending those overlapping tiles together after they have been altered. This module is also fully Autograd & JIT / TorchScript compatible.

This tiling solution is intended for situations where one wishes to render / generate outputs that are larger than what their computing device can support. Tiles can be separately rendered and periodically blended together to maintain tile feature coherence.

## Setup:

**Installation Requirements**
- Python >= 3.6
- PyTorch >= 1.6

**Installation via `pip`:**

```
pip install blended-tiling
```

**Dev / Manual install**:

```
git clone https://github.com/progamergov/blended-tiling.git
cd blended-tiling
pip install -e .

# Notebook installs also require appending to environment variables
# import sys
# sys.path.append('/content/blended-tiling')
```


## Documentation

### `TilingModule`

The base blended tiling module.

```
blended_tiling.TilingModule(tile_size=(224, 224), tile_overlap=(0.25, 0.25), base_size=(512, 512))
```

**Initialization Variables**

* `tile_size` (int or tuple of int): The size of tiles to use. A single integer to use for both the height and width dimensions, or a list / tuple of dimensions with a shape of: `[height, width]`.
* `tile_overlap` (int or tuple of int): The amount of overlap to use when creating tiles. A single integer to use for both the height and width dimensions, or a list / tuple of dimensions with a shape of: `[height, width]`.
* `base_size` (int or tuple of int): The size of the NCHW tensor being split into tiles. A single integer to use for both the height and width dimensions, or a list / tuple of dimensions with a shape of: `[height, width]`.


#### Methods


**`num_tiles()`**

  * Returns
    * `num_tiles` (int): The number of tiles that the full image shape is divided into based on specified parameters.

**`tiling_pattern()`**

  * Returns:
    * `pattern` (list of int): The number of tiles per column and number of tiles per row, in the format of: `[n_tiles_per_column, n_tiles_per_row]`.

**`split_into_tiles(x)`**: Splits an NCHW image input into overlapping tiles, and then returns the tiles. The `base_size` parameter is automatically readjusted to match the input.
  * Returns:
    * `tiles` (torch.Tensor): A set of tiles created from the input image.

**`get_tile_masks(channels=3, device=torch.device("cpu"))`**: Return a stack of NCHW masks corresponding to the tiles outputted by `.split_into_tiles(x)`.

  * Variables:
    * `channels` (int, optional): The number of channels to use for the masks. Default: 3
    * `device` (torch.device, optional): The desired device to create the masks on. Default: torch.device("cpu")
  * Returns:
    * `masks` (torch.Tensor): A set of tile masks stacked across the batch dimension.


**`rebuild(tiles, border=None, colors=None)`**: Creates and returns the full image from a stack of NCHW tiles stacked across the batch dimension.
  * Variables:
    * `tiles` (torch.Tensor): A set of tiles that may or not be masked, stacked across the batch dimension.
    * `border` (int, optional): Optionally add a border of a specified size to the edges of tiles in the full image for debugging and explainability. Set to None for no border.
    * `colors` (list of float, optional): A set of floats to use for the border color, if using borders. Default is set to red unless specified.
  * Returns:
    * `full_image` (torch.Tensor): The full image made up of tiles merged together without any blending.

**`rebuild_with_masks(tiles, border=None, colors=None)`:** Creates and returns the full image from a stack of NCHW tiles stacked across the batch dimension, using tile blend masks.
  * Variables:
    * `tiles` (torch.Tensor): A set of tiles that may or not be masked, stacked across the batch dimension.
    * `border` (int, optional): Optionally add a border of a specified size to the edges of tiles in the full image for debugging and explainability. Set to None for no border.
    * `colors` (list of float, optional): A set of floats to use for the border color, if using borders. Default is set to red unless specified.
  * Returns:
    * `full_image` (torch.Tensor): The full image made up of tiles blended together using masks.

**`forward(x)`:** Takes a stack of tiles, combines them into the full image with blending masks, then splits the image back into tiles.
  * Variables:
    * `x` (torch.Tensor): A set of tiles to blend the overlapping regions together of.
  * Returns:
    * `x` (torch.Tensor): A set of tiles with overlapping regions blended together.


## Usage

The `TilingModule` class is pretty easy to use.

```
from blended_tiling import TilingModule


full_size = [512, 512]
tile_size = [224, 224]
tile_overlap = [0.25, 0.25]  # 25% overlap on both H & W

tiling_module = TilingModule(
    tile_size=tile_size,
    tile_overlap=tile_overlap,
    base_size=full_size,
)

# Shape of tiles expected in forward pass
input_shape = [tiling_module.num_tiles(), 3] + tile_size

# Tiles are blended together and then split apart by default
blended_tiles = tiling_module(torch.ones(input_shape))
```


Tiles can be created and then merged back into the original tensor like this:

```
full_tensor = torch.ones(1, 3, 512, 512)

tiles = tiling_module.split_into_tiles(full_tensor)

full_tensor = tiling_module.rebuild_with_masks(tiles)
```

The tile boundaries can be viewed on the full tensor like this:

```
tiles = torch.ones(9, 3, 224, 224)
full_tensor = tiling_module.rebuild_with_masks(tiles, border=2)
```

And the number of tiles and tiling pattern can be obtained like this:

```
num_tiles = tiling_module.num_tiles()

tiling_pattern = tiling_module.tiling_pattern()
print("{}x{}".format(tiling_pattern[0], tiling_pattern[1]))
```



## Examples


To demonstrate the tile blending abilities of the `TilingModule` class, an example has been created below.



First we'll create a set of tiles & give them all unique colors for this example:

```
# Setup TilingModule instance
full_size = [768, 1014]
tile_size = [256, 448]
tile_overlap = [0.25, 0.25]
tiling_module = TilingModule(
    tile_size=tile_size,
    tile_overlap=tile_overlap,
    base_size=full_size,
)

# Create unique colors for tiles
tile_colors = [
    [0.5334, 0.0, 0.8459],
    [0.0, 1.0, 0.0],
    [0.0, 0.7071, 0.7071],
    [0.7071, 0.7071, 0.0],
    [1.0, 0.0, 0.0],
    [0.8459, 0.0, 0.5334],
    [0.7071, 0.0, 0.7071],
    [0.0, 0.8459, 0.5334],
    [0.5334, 0.8459, 0.0],
    [0.0, 0.5334, 0.8459],
    [0.0, 0.0, 1.0],
    [0.8459, 0.5334, 0.0],
]
tile_colors = torch.as_tensor(tile_colors).view(12, 3, 1, 1)

# Create tiles
tiles = torch.ones([tiling_module.num_tiles(), 3] + tile_size)

# Color tiles
tiles = tiles * tile_colors
```

<img src="https://github.com/ProGamerGov/blended-tiling/raw/main/examples/without_masks_separate_tiles.jpg" width="500">

Next we apply the blend masks to the tiles:

```
tiles = tiles * tiling_module.get_tile_masks()
```

<img src="https://github.com/ProGamerGov/blended-tiling/raw/main/examples/with_masks_separate_tiles.jpg" width="500">


We can now combine the masked tiles into the full image:

```
# Build full tiled image
output = tiling_module.rebuild(tiles)
```

<img src="https://github.com/ProGamerGov/blended-tiling/raw/main/examples/with_masks.jpg" width="500">

We can also view the tile boundaries like so:

```
# Build full tiled image
output = tiling_module.rebuild(tiles, border=2, colors=[0,0,0])
```

<img src="https://github.com/ProGamerGov/blended-tiling/raw/main/examples/with_masks_and_borders.jpg" width="500">

We can view an animation of the tiles being added like this:

```
from torchvision.transforms import ToPILImage

tile_steps = [
    tiling_module.rebuild(tiles[: i + 1]) for i in range(tiles.shape[0])
]
tile_frames = [
    ToPILImage()(x[0])
    for x in [torch.zeros_like(tile_steps[0])] + tile_steps + [tile_steps[-1]]
]
tile_frames[0].save(
    "tiles.gif",
    format="GIF",
    append_images=tile_frames[1:],
    save_all=True,
    duration=700,
    loop=0,
)
```

<img src="https://github.com/ProGamerGov/blended-tiling/raw/main/examples/with_masks.gif" width="500">
