# %%
import numpy as np
import plotly.graph_objects as go

# %%
tt = np.linspace(0,1000,6)-100
tt[0] = 0
SmallMU_times= np.array(
    [150,258,325,457,500,555,641,687,700,742,759,830,875,919,935,975]
    )
LargeMU_times = np.array([509,686,722,848,890,958])
SmallMU_rates = np.array([0,1,2,3,5,5])
LargeMU_rates = np.array([0,0,0,1,2,3])

fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=SmallMU_times,
#     y=np.ones(len(SmallMU_times))*1.45*max(SmallMU_rates),
#     name="Small MU Spikes",
#     showlegend=False,
#     mode="markers",
#     marker_symbol="line-ns-open",
#     marker_line_width=4,
#     marker_size=15,
#     marker_color="skyblue", #'rgba(80,200,200,1)',
#     line_color="rgba(20,20,20,1)",
# ))

# fig.add_trace(go.Scatter(
#     x=LargeMU_times,
#     y=np.ones(len(LargeMU_times))*1.3*max(SmallMU_rates),
#     name="Large MU Spikes",
#     showlegend=False,
#     mode="markers",
#     marker_symbol="line-ns-open",
#     marker_line_width=4,
#     marker_size=15,
#     marker_color="firebrick", #'rgba(80,200,200,1)',
#     line_color="rgba(20,20,20,1)",
# ))

fig.add_trace(go.Scatter(
    x=SmallMU_rates,
    y=LargeMU_rates,
    name="MU Population <br>Trajectory   ",
    showlegend=True,
    mode="lines+markers",
    marker_line_width=4,
    marker_line_color="black",
    marker_size=15,
    marker_color="black", #'rgba(80,200,200,1)',
    line_color="black",
    line_width=4
))

# fig.add_trace(go.Scatter(
#     x=tt,
#     y=LargeMU_rates,
#     name="Large MU",
#     mode="lines+markers",
#     marker_line_width=2,
#     marker_size=15,
#     marker_color="firebrick",#'rgba(80,200,200,1)',
#     line_color="rgba(30,30,30,1)",
# ))

# fig.add_trace(go.Scatter(
#     x=[-100,1100],
#     y=[5.8,5.8],
#     showlegend=False,
#     marker_symbol="line-ew",
#     line_color="rgba(30,30,30,.1)",
# ))

fig.update_xaxes(
    title_text="Small MU",
    title_font_color="skyblue",
    ticklabelposition="inside right",
    gridcolor="rgba(30,30,30,.1)",
    zerolinecolor="firebrick",
    tickvals=np.arange(6))

fig.update_yaxes(
    title_text="Large MU",
    title_font_color="firebrick",
    ticklabelposition="inside top",
    gridcolor="rgba(30,30,30,.1)",
    zerolinecolor="skyblue",
    tickvals=np.arange(4),
    title_standoff=5)

fig.update_layout(
    #title=dict(text="",x=0.5,y=.88),
    font_family="Arial",
    font_size=24,
    font_color="black",
    legend=dict(
        x=.09,
        y=.54,
        bgcolor="rgba(231,211,179,1)", # manila
    ),
    width=800,
    height=600,
    plot_bgcolor="rgba(231,211,179,1)", # manila
    )

fig.show()
# %%
fig.write_image('./figs/fig1b.svg')

# %%
