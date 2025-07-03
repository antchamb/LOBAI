#
# import dash
# from dash import html, dcc, Input, Output, callback
# import plotly.graph_objects as go
# import plotly.express as px
# import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# from models.deeplob import Deeplob
# from data_loader.FI2010 import Dataset_fi2010
# from captum.attr import IntegratedGradients
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import random
#
# dash.register_page(__name__, name='deeplob', path='/deeplob')
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# net = Deeplob(lighten=True).to(device)
# net.load_state_dict(torch.load("weights/deeplob_light.pt", map_location=device))
# net.eval()
#
# ds = Dataset_fi2010(
#     auction=False,
#     normalization="Zscore",
#     stock_idx=[0, 1, 2, 3, 4],  # all five stocks
#     days=[2],  # day 1 = training set; 2-10 = test sets
#     T=20,
#     k=0,
#     lighten=True,
# )
# layout = html.Div([
#
# ])