"""Utils for Multi-objective loop."""

import plotly.graph_objects as go


def _plot_scatter_plot(scores_pred, scores_gt, name=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores_gt, y=scores_pred, mode="markers", name=f"{name}", marker=dict(size=5)))
    fig.update_layout(
        title="Scatter Plot of Predicted vs Ground Truth Scores",
        xaxis_title="Ground Truth Scores",
        yaxis_title="Predicted Scores",
        width=600,
        height=500,
    )
    return fig
