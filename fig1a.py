# %%
import numpy as np
import plotly.graph_objects as go

# %%
tt = np.linspace(0,1000,6)-100
tt[0] = 0
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
    marker_line_width=6,
    marker_size=15,
    marker_color="skyblue", #'rgba(80,200,200,1)',
    line_color="black",
))

fig.add_trace(go.Scatter(
    x=LargeMU_times,
    y=np.ones(len(LargeMU_times))*1.3*max(SmallMU_rates),
    name="Large MU Spikes",
    showlegend=False,
    mode="markers",
    marker_symbol="line-ns-open",
    marker_line_width=6,
    marker_size=15,
    marker_color="firebrick", #'rgba(80,200,200,1)',
    line_color="black",
))

# fig.add_trace(go.Scatter(
#     x=[0,1000],
#     y=[9.5,11.5],
#     name="Force  ",
#     mode="lines",
#     line_color="black",
#     line_width=4
# ))

fig.add_trace(go.Scatter(
    x=tt,
    y=SmallMU_rates,
    name="Small MU  ",
    mode="lines+markers",
    marker_line_width=4,
    marker_line_color="skyblue",
    marker_size=15,
    marker_color="skyblue", #'rgba(80,200,200,1)',
    line_color="skyblue",
    line_width=4
))

fig.add_trace(go.Scatter(
    x=tt,
    y=LargeMU_rates,
    name="Large MU  ",
    mode="lines+markers",
    marker_line_width=4,
    marker_line_color="firebrick",
    marker_size=15,
    marker_color="firebrick",#'rgba(80,200,200,1)',
    line_color="firebrick",
    line_width=4
))

fig.add_trace(go.Scatter(
    x=[-100,1100],
    y=[5.8,5.8],
    showlegend=False,
    marker_symbol="line-ew",
    line_color="rgba(30,30,30,.1)",
))

fig.update_xaxes(title_text="Time (a.u.)",showticklabels=False,gridcolor="rgba(30,30,30,.1)",zerolinecolor="rgba(30,30,30,.1)",range=[-100,1100])
fig.update_yaxes(title_text="Spikes",ticklabelposition="inside top",gridcolor="rgba(30,30,30,.1)",zerolinecolor="rgba(30,30,30,.1)",tickvals=np.arange(6),title_standoff=5,automargin=False)

fig.update_layout(
    #title=dict(text="",x=0.5,y=.88),
    font_family="Arial",
    font_size=24,
    font_color="black",
    legend=dict(
        x=.09,
        y=.54,
        bgcolor="rgba(231,211,179,1)"
    ),
    autosize=False,
    width=800,
    height=600,
    plot_bgcolor="rgba(231,211,179,1)", # manila
    )

fig.show()
# %%
fig.write_image('./figs/fig1a_.svg')

# %%


# %%
