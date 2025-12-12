import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go


def main():
    # 1. Random points
    pts = np.random.rand(200, 2)

    # 2. Delaunay triangulation
    tri = Delaunay(pts)

    # 4 colours to choose from
    palette = ["#fece8b", "#b49370", "#95CC84", "#f7b4c3"]

    fig = go.Figure()

    # For each triangle
    for simplex in tri.simplices:
        vertices = pts[simplex]
        x = vertices[:, 0]
        y = vertices[:, 1]

        # close the polygon
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                mode="lines",
                line=dict(color="white", width=6),
                fillcolor=np.random.choice(palette),
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
    _ = fig.update_xaxes(range=[0, 1])
    _ = fig.update_yaxes(range=[0, 1])

    fig.write_image("triangulation.svg")  # pyright: ignore[reportArgumentType]
    fig.write_image("triangulation.png")  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    main()
