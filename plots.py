# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:41:31 2024
@author: vitor
"""

import pandas as pd
import plotly.express as px

results_path = "results/Effisegnet results.xlsx"

results = pd.read_excel(results_path)

results = results.rename(columns={'test_loss': "Loss", 'test_accuracy': "Accuracy",
                                  'test_dice': "Dice", 'test_iou': "IoU",
                                  'test_recall': "Recall", 'test_precision': "Precision",
                                  'test_f1':"F1-score"})

#%% Violin Plot Example

# df = px.data.tips()
# fig = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all",
#           hover_data=df.columns)
# fig.show()
# fig.write_image("fig1.svg")

#%% Violin Plot #1: All

df = pd.melt(results, id_vars=["N fold"], var_name="Metric Name", value_name="Metric Value")

fig = px.violin(df, y="Metric Value", x="Metric Name", color="Metric Name", box=True, points="all",
          hover_data=df.columns, title="Resultados Effisegnet K-fold k=10")

fig.show()
fig.write_image("results/fig1.svg", width=1000, height=500)
fig.write_image("results/fig1.png", width=1000, height=500)

#%% Violin Plot #2: Not Loss

df = df[df["Metric Name"]!="Loss"]
df = df[df["Metric Name"]!="F1-score"]
fig = px.violin(df, y="Metric Value", x="Metric Name", color="Metric Name",
                box=True, points="all", hover_data=df.columns,  title="Resultados Effisegnet K-fold k=10")

fig.show()
fig.write_image("results/fig2.svg", width=1000, height=500)
fig.write_image("results/fig2.png", width=1000, height=500)
