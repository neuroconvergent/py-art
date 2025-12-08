> [!INFO]
> This README is almost completely AI generated because I am lazy.

# py-art

py-art is a Python project for generating geometric artwork—specifically randomized Delaunay-triangulated patterns—designed for high-resolution digital printing on glass doors and panels. The project uses Poisson-disc sampling and Delaunay triangulation to create organic, non-repeating geometric structures with customizable colours, sizing, and DPI.

## Features

- Generates geometric artwork using:
  - Bridson’s Poisson-disc sampling (evenly spaced random points)
  - Delaunay triangulation
  - Anti-clumping colour assignment
- Thick white triangle borders for a stained-glass effect
- Exports high-resolution SVG suitable for large-format printing
- Fully parametric: DPI, inches, spacing, palette, etc.
- Automatically generates multiple unique art pieces

## How It Works

1. A Poisson-disc point set is generated to control minimum spacing.
2. A Delaunay triangulation is computed. The domain for triangulation is larger than the generated image to remove gaps in edges.
3. Each triangle is coloured with anti-clumping logic so neighbours rarely share colours.
4. Plotly renders the final artwork and exports an SVG at print-ready resolution.

## Installation

Clone the repository:

```bash
    git clone --depth 1 https://github.com/neuroconvergent/py-art
    cd py-art
```

Install dependencies in a virtual environment using `uv` (see [PEP 668](https://peps.python.org/pep-0668/)):

```bash
    uv venv && uv sync
```

## Usage

Run the generator:

```bash
    uv run main.py
```

This will produce SVG files such as:

```bash
    triangular_pattern_1.svg
    triangular_pattern_2.svg
    ...
    triangular_pattern_9.svg
```

To customize the artwork, modify the create_image() parameters inside main.py:

```bash
    create_image(
        name="triangular_pattern_1",
        dpi=96,
        width_inch=20,
        height_inch=60,
        border_width_inch=1
    )
```

You may adjust:

- Panel dimensions (inches)
- DPI
- Minimum triangle size
- Colour palette
- Border thickness
- Number of generated files

The output SVG matches the target print size precisely.

## Output Format

- SVG vector graphics
- Perfectly scalable with no loss of detail
- Optimized for glass printing
- Thick white borders enhance colour separation and readability through translucent materials

## License

This project is licensed under the BSD 2-Clause License.
See the [LICENSE](LICENSE) file for details.
