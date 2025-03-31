# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:41:31 2024
@author: vitor
"""

import pandas as pd
import plotly.express as px

results_path = "results/Effisegnet results.xlsx"

results = pd.read_excel(results_path, sheet_name="v3")

results = results.rename(columns={'test_loss': "Loss", 'test_accuracy': "Accuracy", 'test_iou': "IoU",
                                  'test_recall': "Recall", 'test_precision': "Precision", 'test_f1':"Dice"})

df = pd.melt(results, id_vars=["N fold"], var_name="Metric Name", value_name="Metric Value")
 
#%% Violin Plot: Metrics

df = df[df["Metric Name"]!="Loss"]
fig = px.violin(df, y="Metric Value", x="Metric Name", color="Metric Name",
                box=True, points="all", hover_data=df.columns, #title="Resultados Effisegnet K-fold k=10"
)

fig.update_layout(
    showlegend=False,
    yaxis_title=None,
    xaxis_title=None,
    font=dict(
        family="Arial Black",
        size=20
    )
)

fig.show()
fig.write_image("results/fig5.svg", width=1000, height=500)
fig.write_image("results/fig5.png", width=1200, height=600)
