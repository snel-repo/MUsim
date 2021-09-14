import numpy as np
import plotly.graph_objects as go

# %%
tt = np.linspace(0,1000,6)-100
tt[0] = 0
SmallMU_times= np.array([150,258,325,457,500,555,641,687,700,742,759,830,875,919,935,975])
LargeMU_times = np.array([509,686,722,848,890,958])
SmallMU_rates1 = np.array([0,2,3,5,5])
MedMU_rates1 =   np.array([0,1,2,3,3])
LargeMU_rates1 = np.array([0,0,1,2,3])

SmallMU_rates2 = np.array([0,0,2,4,5])
MedMU_rates2 =   np.array([0,1,2,3,3])
LargeMU_rates2 = np.array([0,2,4,4,4])

# SmallMU_rates3 = np.array([0,1,1,1,1,2])
# MedMU_rates3 =   np.array([0,2,3,3,4,4])
# LargeMU_rates3 = np.array([0,1,2,2,3,5])

# %%
fig = go.Figure(go.Scatter3d())

fig.add_scatter3d(
    x=[0,5],
    y=[0,0],
    z=[0,0],
    name="x-axis",
    showlegend=False,
    mode='lines',
    line_color="rgba(100,70,40,1)",
    line_width=3,
    line_dash='dash'
)
fig.add_scatter3d(
    x=[0,0],
    y=[0,5],
    z=[0,0],
    name="y-axis",
    showlegend=False,
    mode='lines',
    line_color="rgba(100,70,40,1)",
    line_width=3,
    line_dash='dash'
)
fig.add_scatter3d(
    x=[0,0],
    y=[0,0],
    z=[0,5],
    name="z-axis",
    showlegend=False,
    mode='lines',
    line_color="rgba(100,70,40,1)",
    line_width=3,
    line_dash='dash'
)
fig.add_scatter3d(
    x=MedMU_rates2,
    y=SmallMU_rates2,
    z=LargeMU_rates2,
    name="condition 2",
    showlegend=True,
    marker_line_width=6,
    marker_line_color="white",
    marker_size=5,
    marker_color="white", #'rgba(80,200,200,1)',
    line_color="white",
    line_width=5
    )

fig.add_scatter3d(
    x=MedMU_rates1,
    y=SmallMU_rates1,
    z=LargeMU_rates1,
    name="condition 1",
    showlegend=True,
    marker_line_width=6,
    marker_line_color="black",
    marker_size=5,
    marker_color="black", #'rgba(80,200,200,1)',
    line_color="black",
    line_width=5
)

fig.update_scenes(
    xaxis_title="Medium-Threshold MU",
    yaxis_title="Low-Threshold MU",
    zaxis_title="High-Threshold MU",
    xaxis_color="green",
    yaxis_color="darkblue",
    zaxis_color="firebrick",
    xaxis_dtick=1,
    yaxis_dtick=1,
    zaxis_dtick=1,
    xaxis_tickangle=0,
    yaxis_tickangle=0,
    zaxis_tickangle=0,
    xaxis_tickfont_size=16,
    yaxis_tickfont_size=16,
    zaxis_tickfont_size=16,
    aspectmode="manual",
    xaxis_backgroundcolor="rgba(240,210,180,1)",
    yaxis_backgroundcolor="rgba(240,210,180,1)",
    zaxis_backgroundcolor="rgba(240,210,180,1)",
    xaxis_zerolinecolor="rgba(240,210,180,1)",
    yaxis_zerolinecolor="rgba(240,210,180,1)",
    zaxis_zerolinecolor="rgba(240,210,180,1)",
    xaxis_zerolinewidth=3,
    yaxis_zerolinewidth=3,
    zaxis_zerolinewidth=3,
    xaxis_gridcolor="rgba(230,200,170,1)",
    yaxis_gridcolor="rgba(230,200,170,1)",
    zaxis_gridcolor="rgba(230,200,170,1)",
    xaxis_title_font_color="green",
    yaxis_title_font_color="darkblue",
    zaxis_title_font_color="firebrick")

fig.update_layout(
    font_family="Arial",
    font_size=18,
    font_color="black",
    legend=dict(
        x=.55,
        y=.73,
        bgcolor="rgba(240,210,180,1)", # manila
        traceorder="reversed",
    ),
    margin=dict(l=0, r=0, b=0, t=0), # tight layout
    scene=dict(
        camera=dict(
            eye=dict(
                x=1,y=.7,z=.5
            ),
            projection=dict(
                type="orthographic"
            )    
        )
    ),
    width=800,
    height=600,
)

fig.show()

# %%
fig.write_image('./figs/fig1c.pdf')