# ---------------------------------------------------------------------------------------
# Licensed under BSD 2-clause https://github.com/neuroconvergent/py-art/blob/main/LICENSE
# ---------------------------------------------------------------------------------------
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay


def main():
    for i in range(1, 10):
        create_image(
            name=f"triangular_pattern_{i}",
            dpi=96,
            width_inch=20,
            height_inch=60,
            border_width_inch=1,
        )


def poisson_disc_samples(width, height, r, k=30):
    """
    Bridson's Poisson-disc sampling algorithm.

    width, height : sampling area
    r             : minimum distance between samples
    k             : number of candidate attempts per active point
    """
    cell_size = r / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    # Grid to store sample indices (-1 means empty)
    grid = -np.ones((grid_height, grid_width), dtype=int)

    samples = []
    active = []

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

            # Check neighbouring cells for conflicts
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


def create_image(
    name: str,
    dpi: int,
    width_inch: float,
    height_inch: float,
    border_width_inch: float,
    palette: list[str] = ["#fece8b", "#b49370", "#95CC84", "#f7b4c3"],
) -> None:
    # Calculate resolution for printed glass
    width = int(width_inch * dpi)
    height = int(height_inch * dpi)

    x_scale = 1.0
    y_scale = height / width
    x_range = 3 * x_scale
    y_range = 3 * y_scale
    x_mid = x_range / 2
    y_mid = y_range / 2

    # Calculate width of white border in pixels
    border_width_px = int(border_width_inch * dpi / 2)  # border on both sides

    # Define minimum spacing between points
    minimum_triangle_size_inch = 5
    poisson_x_scale = width_inch / x_range
    poisson_y_scale = height_inch / y_range
    min_scale = min(poisson_x_scale, poisson_y_scale)

    # Random points
    r = minimum_triangle_size_inch / (2 * min_scale)  # minimum spacing
    pts = poisson_disc_samples(width=x_range, height=y_range, r=r)
    pts[:, 0] -= x_mid
    pts[:, 1] -= y_mid

    # OPTIONAL: Add jitter to increase randomness while keeping overall spacing
    # jitter_strength = r * 0.65  # ~35% of minimum spacing
    # pts += np.random.uniform(-jitter_strength, jitter_strength, pts.shape)

    # Delaunay triangulation
    tri = Delaunay(pts)

    # 4 colors to choose from
    triangle_colors = [None] * len(tri.simplices)

    # Precompute adjacency list
    adj = [[] for _ in tri.simplices]
    edges = {}

    for i, s in enumerate(tri.simplices):
        for a, b in [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]:
            key = tuple(sorted((a, b)))
            if key in edges:
                j = edges[key]
                adj[i].append(j)
                adj[j].append(i)
            else:
                edges[key] = i

    # Assign colors with anti-clumping
    for i in range(len(tri.simplices)):
        neighbors = adj[i]
        disallowed = {
            triangle_colors[n] for n in neighbors if triangle_colors[n] is not None
        }

        # Retry random colors until one fits
        for _ in range(10):
            c = np.random.choice(palette)
            if c not in disallowed:
                triangle_colors[i] = c
                break
        else:
            # fallback (should almost never happen)
            triangle_colors[i] = np.random.choice(palette)

    fig = go.Figure()

    # For each triangle
    for i, simplex in enumerate(tri.simplices):
        vertices = pts[simplex]
        x = vertices[:, 0]
        y = vertices[:, 1]
        color = triangle_colors[i]

        # close the polygon
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                mode="lines",
                line=dict(color="white", width=border_width_px),  # ✔ thick white border
                fillcolor=color,
            )
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 3])

    fig.write_image(f"{name}.svg", width=width, height=height)


if __name__ == "__main__":
    main()
