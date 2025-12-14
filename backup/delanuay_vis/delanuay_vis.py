# Copyright 2025 Sundar Gurumurthy
# SPDX-License-Identifier: BSD-2-License

"""
Script to visualize Delaunay triangulation with circumcircles.

This script generates Poisson-disc sampled points within a specified area,
computes the Delaunay triangulation, and creates a visualization showing
the triangles along with their circumcircles.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from enum import Enum
import os


def main() -> None:
    """
    Main function to generate and visualize Delaunay triangulation with circumcircles.

    Generates points using Poisson-disc sampling, computes Delaunay triangles,
    plots the triangles and their circumcircles using Plotly, and saves the
    visualization in multiple formats (PNG, SVG by default).
    """
    root_dir: str = "images"
    os.makedirs(root_dir, exist_ok=True)

    width = 1
    height = 1
    min_distance = 0.4
    points = poisson_disc_samples(width, height, min_distance, seed=300)

    triangles, adj = delaunay_triangulation(points)

    fig = go.Figure()
    for triangle_coords in triangles:
        # Triangle vertices
        vertex_x = triangle_coords[:, 0]
        vertex_y = triangle_coords[:, 1]
        # Close the triangle
        vertex_x = np.append(vertex_x, vertex_x[0])
        vertex_y = np.append(vertex_y, vertex_y[0])

        fig.add_trace(
            go.Scatter(
                x=vertex_x,
                y=vertex_y,
                mode="lines",
                line=dict(color="black", width=4),
                showlegend=False,
            )
        )

        radius, centre = circumcircle(triangle_coords)

        fig.add_shape(
            type="circle",
            x0=centre[0] - radius,
            y0=centre[1] - radius,
            x1=centre[0] + radius,
            y1=centre[1] + radius,
            line=dict(color="black", width=2, dash="dash"),
        )

    _ = fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    output_path = os.path.join(root_dir, "delanuay")
    formats = output_formats()
    save_plot(fig, output_path, formats, image_height=800, image_width=800)


class OutputFormat(Enum):
    HTML = 0
    PNG = 1
    SVG = 2


def output_formats(
    html: bool = False, png: bool = True, svg: bool = True
) -> list[OutputFormat]:
    """
    Create a list of output formats for saving plots.

    Parameters
    ----------
    html : bool, optional
        Whether to include HTML format, by default False.
    png : bool, optional
        Whether to include PNG format, by default True.
    svg : bool, optional
        Whether to include SVG format, by default True.

    Returns
    -------
    list[OutputFormat]
        List of selected output formats.
    """
    output_formats: list[OutputFormat] = []
    if html:
        output_formats.append(OutputFormat.HTML)
    if png:
        output_formats.append(OutputFormat.PNG)
    if svg:
        output_formats.append(OutputFormat.SVG)

    return output_formats


def save_plot(
    fig: go.Figure,
    base_path: str,
    formats: list[OutputFormat],
    image_height: int = 1080,
    image_width: int = 1920,
) -> None:
    """
    Save the Plotly figure in specified formats.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to save.
    base_path : str
        Base path for saving files (without extension).
    formats : list[OutputFormat]
        List of formats to save in.
    image_height : int, optional
        Height of the image for PNG and SVG, by default 1080.
    image_width : int, optional
        Width of the image for PNG and SVG, by default 1920.
    """
    for output_format in formats:
        if output_format is OutputFormat.HTML:
            fig.write_html(base_path + ".html")
        if output_format is OutputFormat.SVG:
            fig.write_image(base_path + ".svg", height=image_height, width=image_width)
        if output_format is OutputFormat.PNG:
            fig.write_image(base_path + ".png", height=image_height, width=image_width)


def poisson_disc_samples(
    width: float, height: float, r: float, k: int = 30, seed: int | None = None
) -> np.ndarray:
    """
    Bridson's Poisson-disc sampling algorithm.

    Parameters
    ----------
    width : float
        sampling area width
    height : float
        sampling area height
    r : float
        minimum distance between samples
    k : int, optional
        number of candidate attempts per active point, by default 30
    seed : int, optional
        random seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing the generated sample points.
    """
    cell_size = r / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    # Grid to store sample indices (-1 means empty)
    grid = -np.ones((grid_height, grid_width), dtype=int)

    samples: list[np.ndarray] = []
    active: list[int] = []

    if seed is not None:
        np.random.seed(seed)

    # Start with a random point
    p = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
    samples.append(p)
    active.append(0)

    gx = int(p[0] // cell_size)
    gy = int(p[1] // cell_size)
    grid[gy, gx] = 0

    while active:
        idx = np.random.choice(active)
        base = samples[idx]
        found = False

        # Try k random points around `base`
        for _ in range(k):
            theta = np.random.uniform(0, 2 * np.pi)
            rad = np.random.uniform(r, 2 * r)
            candidate = base + rad * np.array([np.cos(theta), np.sin(theta)])

            # Discard if outside the domain
            if not (0 <= candidate[0] < width and 0 <= candidate[1] < height):
                continue

            # Check 5x5 neighbourhood of cells for conflicts (ensures min distance)
            cgx = int(candidate[0] // cell_size)
            cgy = int(candidate[1] // cell_size)

            ok = True
            for yy in range(max(0, cgy - 2), min(grid_height, cgy + 3)):
                for xx in range(max(0, cgx - 2), min(grid_width, cgx + 3)):
                    si = grid[yy, xx]
                    if si != -1:
                        if np.linalg.norm(samples[si] - candidate) < r:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                samples.append(candidate)
                active.append(len(samples) - 1)
                grid[cgy, cgx] = len(samples) - 1
                found = True
                break

        if not found:
            active.remove(idx)

    return np.array(samples)


def delaunay_triangulation(
    points: np.ndarray,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """
    Compute Delaunay triangulation from a set of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing the points.

    Returns
    -------
    tuple[list[np.ndarray], list[list[int]]]
        List of triangles, each as an array of shape (3, 2) with the vertices,
        and list of lists of neighbouring triangle indices.
    """
    if len(points) < 3:
        return [], []

    tri = Delaunay(points)
    triangles = []
    for simplex in tri.simplices:
        triangle = points[simplex]
        triangles.append(triangle)

    neighbors = tri.neighbors.tolist()
    return triangles, neighbors


def circumcircle(tri: np.ndarray) -> tuple[float, tuple[float, float]]:
    """
    Return radius, centre of circumcircle of a given triangle
    """
    if len(tri) != 3:
        raise Exception("Triangle has more than 3 points")

    A, B, C = tri[0], tri[1], tri[2]
    A_x, A_y = A
    B_x, B_y = B
    C_x, C_y = C

    D = 2 * (A_x * (B_y - C_y) + B_x * (C_y - A_y) + C_x * (A_y - B_y))

    if D == 0:
        raise Exception("Points are collinear, no circumcircle")

    center_x = (
        (A_x**2 + A_y**2) * (B_y - C_y)
        + (B_x**2 + B_y**2) * (C_y - A_y)
        + (C_x**2 + C_y**2) * (A_y - B_y)
    ) / D

    center_y = (
        (A_x**2 + A_y**2) * (C_x - B_x)
        + (B_x**2 + B_y**2) * (A_x - C_x)
        + (C_x**2 + C_y**2) * (B_x - A_x)
    ) / D

    radius = np.sqrt((center_x - A_x) ** 2 + (center_y - A_y) ** 2)

    return radius, (center_x, center_y)


if __name__ == "__main__":
    main()
