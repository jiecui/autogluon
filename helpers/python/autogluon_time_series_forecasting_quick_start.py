# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: autogluon
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AutoGluon Time Series - Forecasting Quick Start

# %% [markdown]
# ## Loading time series data as a ```TimeSeriesDataFrame```

# %% [markdown]
# ### Import required modules

# %%
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# %% [markdown]
# ### Load a subset of the M4 hourly dataset as a ```pandas.DataFrame```

# %%
df = pd.read_csv(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv"
)
df.head()

# %%
train_data = TimeSeriesDataFrame.from_data_frame(
    df, id_column="item_id", timestamp_column="timestamp"
)
train_data.head()
