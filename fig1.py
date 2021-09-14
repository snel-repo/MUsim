import numpy as np
import plotly.io as pio
pio.orca.config.use_xvfb = True
import plotly.graph_objects as go
from MUsim import MUsim

# %%
tt = np.linspace(0,1000,6)
SmallMU_times= np.array([150,258,325,457,500,555,641,687,700,742,759,830,875,919,935,975])
LargeMU_times = np.array([509,686,722,848,890,958])
SmallMU_rates = np.array([0,1,2,3,5,5])
LargeMU_rates = np.array([0,0,0,1,2,3])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=SmallMU_times,
    y=np.ones(len(SmallMU_times))*1.45*max(SmallMU_rates),
    name="Small MU Spikes",
    showlegend=False,
    mode="markers",
    marker_symbol="line-ns-open",
    marker_line_width=4,
    marker_size=15,
    marker_color="skyblue", #'rgba(80,200,200,1)',
    line_color="rgba(20,20,20,1)",
))

fig.add_trace(go.Scatter(
    x=LargeMU_times,
    y=np.ones(len(LargeMU_times))*1.3*max(SmallMU_rates),
    name="Large MU Spikes",
    showlegend=False,
    mode="markers",
    marker_symbol="line-ns-open",
    marker_line_width=4,
    marker_size=15,
    marker_color="firebrick", #'rgba(80,200,200,1)',
    line_color="rgba(20,20,20,1)",
))

fig.add_trace(go.Scatter(
    x=tt,
    y=SmallMU_rates,
    name="Small MU",
    mode="lines+markers",
    marker_line_width=2,
    marker_size=15,
    marker_color="skyblue", #'rgba(80,200,200,1)',
    line_color="rgba(20,20,20,1)",
))

fig.add_trace(go.Scatter(
    x=tt,
    y=LargeMU_rates,
    name="Large MU",
    mode="lines+markers",
    marker_line_width=2,
    marker_size=15,
    marker_color="firebrick",#'rgba(80,200,200,1)',
    line_color="rgba(20,20,20,1)",
))

fig.update_xaxes(title_text='Time (a.u.)',showticklabels=False,gridcolor="rgba(30,30,30,.1)",zerolinecolor="rgba(30,30,30,.1)")
fig.update_yaxes(title_text='Spike Rates                         Spikes',gridcolor="rgba(30,30,30,.1)",zerolinecolor="rgba(30,30,30,.1)", )

fig.update_layout(
    title=dict(text=""),
    plot_bgcolor="rgba(241,221,189,1)", # manila
    )

fig.show()
# %%
# fig.write_image('./figures/figure1.eps')
# %%
