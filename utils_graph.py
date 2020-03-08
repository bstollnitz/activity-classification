from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

COLORS = ['#f4792e', '#24624f', '#c7303b', '#457abf', '#298964', '#ffd769']


def plot_nn_results(train_results: List[float], test_results: List[float], 
    title: str, y_title_test: str, dir_name: str, file_name: str) -> None:
    """
    Plots accuracy or loss of a neural network over epochs, for training and
    test set.
    """
    epochs = list(range(len(train_results)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, 
        y=train_results, 
        line=dict(color=COLORS[0]),
        name='Training'))
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=test_results, 
        line=dict(color=COLORS[1]),
        name='Test'))
    fig.update_xaxes(
        title_text='Epoch')
    fig.update_yaxes(
        title_text=y_title_test)

    fig.update_layout(
        title=title,
        yaxis_rangemode='tozero'
        )

    local_dir_path = Path('.', dir_name)
    local_dir_path.mkdir(exist_ok=True)
    path = Path(local_dir_path, file_name)
    pio.write_html(fig, str(path))
