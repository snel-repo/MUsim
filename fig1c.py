import numpy as np
import plotly.graph_objects as go

# %%
tt = np.linspace(0,1000,6)-100
tt[0] = 0
SmallMU_times= np.array([150,258,325,457,500,555,641,687,700,742,759,830,875,919,935,975])
LargeMU_times = np.array([509,686,722,848,890,958])
SmallMU_rates1 = np.array([0,1,2,3,5,5])
MedMU_rates1 = np.zeros((6,))
LargeMU_rates1 = np.array([0,0,0,1,2,3])

SmallMU_rates2 = np.array([0,0,1,1,2,2])
MedMU_rates2 = np.array([0,1,2,2,4,4])
LargeMU_rates2 = np.array([0,2,3,3,4,5])

SmallMU_rates3 = np.array([0,1,1,1,1,2])
MedMU_rates3 = np.array([0,2,3,3,4,4])
LargeMU_rates3 = np.array([0,1,2,2,3,5])

# %%
fig = go.Figure(go.Scatter3d())

fig.add_scatter3d(
    x=MedMU_rates3,
    y=SmallMU_rates3,
    z=LargeMU_rates3,
    name="context 3",
    showlegend=True,
    marker_line_width=5,
    marker_line_color="lightslategray",
    marker_size=5,
    marker_color="lightslategray", #'rgba(80,200,200,1)',
    line_color="lightslategray",
    line_width=5
    )

fig.add_scatter3d(
    x=MedMU_rates2,
    y=SmallMU_rates2,
    z=LargeMU_rates2,
    name="context 2",
    showlegend=True,
    marker_line_width=5,
    marker_line_color="darkslategrey",
    marker_size=5,
    marker_color="darkslategrey", #'rgba(80,200,200,1)',
    line_color="darkslategrey",
    line_width=5
    )

fig.add_scatter3d(
    x=MedMU_rates1,
    y=SmallMU_rates1,
    z=LargeMU_rates1,
    name="context 1",
    showlegend=True,
    marker_line_width=5,
    marker_line_color="black",
    marker_size=5,
    marker_color="black", #'rgba(80,200,200,1)',
    line_color="black",
    line_width=5
)


# fig.update_xaxes(title_text="Small MU",title_font_color="skyblue",ticklabelposition="inside right",gridcolor="rgba(30,30,30,.1)",zerolinecolor="firebrick",tickvals=np.arange(6))
# fig.update_yaxes(title_text="Large MU",title_font_color="firebrick",ticklabelposition="inside top",gridcolor="rgba(30,30,30,.1)",zerolinecolor="skyblue",tickvals=np.arange(4)

fig.update_scenes(
    xaxis_title="Medium MU",
    yaxis_title="Small MU",
    zaxis_title="Large MU",
    xaxis_color="green",
    yaxis_color="skyblue",
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
    aspectmode="data",
    xaxis_backgroundcolor="rgba(231,211,179,1)",
    yaxis_backgroundcolor="rgba(231,211,179,1)",
    zaxis_backgroundcolor="rgba(231,211,179,1)",
    xaxis_zerolinecolor="rgba(30,30,30,.4)",
    yaxis_zerolinecolor="rgba(30,30,30,.4)",
    zaxis_zerolinecolor="rgba(30,30,30,.4)",
    xaxis_gridcolor="rgba(30,30,30,.4)",
    yaxis_gridcolor="rgba(30,30,30,.4)",
    zaxis_gridcolor="rgba(30,30,30,.4)",
    xaxis_title_font_color="green",
    yaxis_title_font_color="skyblue",
    zaxis_title_font_color="firebrick")

fig.update_layout(
    font_family="Arial",
    font_size=20,
    font_color="black",
    legend=dict(
        x=.5,
        y=.75,
        bgcolor="rgba(231,211,179,1)", # manila
        traceorder="reversed",
    ),
    margin=dict(l=0, r=0, b=0, t=0), # tight layout
    scene=dict(
        camera=dict(
            eye=dict(
                x=1,y=.8,z=.6
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