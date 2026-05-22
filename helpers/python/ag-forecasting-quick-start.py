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
#     display_name: tsf-fm
#     language: python
#     name: python3
# ---

# %% [markdown] id="b0f6df2b"
# # AutoGluon Time Series - Forecasting Quick Start
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/stable/docs/tutorials/timeseries/forecasting-quick-start.ipynb)
# [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/stable/docs/tutorials/timeseries/forecasting-quick-start.ipynb)
#
#
# Via a simple `fit()` call, AutoGluon can train and tune
#
# - simple forecasting models (e.g., ARIMA, ETS, Theta),
# - powerful deep learning models (e.g., DeepAR, Temporal Fusion Transformer),
# - tree-based models (e.g., LightGBM),
# - an ensemble that combines predictions of other models
#
# to produce multi-step ahead _probabilistic_ forecasts for univariate time series data.
#
# This tutorial demonstrates how to quickly start using AutoGluon to generate hourly forecasts for the [M4 forecasting competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) dataset.
#
# ## Loading time series data as a `TimeSeriesDataFrame`
#
# First, we import some required modules

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 54618, "status": "ok", "timestamp": 1761449688961, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="aa00faab-252f-44c9-b8f7-57131aa8251c" outputId="3cf27ae1-72f1-4230-c4ca-22d1853798fa" tags=["remove-cell"]
# We use uv for faster installation
# %pip install uv
# %uv pip install -q autogluon.timeseries --system
# %uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab


# %% [markdown] id="519d689a"
# To use `autogluon.timeseries` , we will only need the following two classes:
#
# * `TimeSeriesDataFrame` stores a dataset consisting of multiple time series.
# * `TimeSeriesPredictor` takes care of fitting, tuning and selecting the best forecasting models, as well as generating new forecasts.
#

# %% executionInfo={"elapsed": 4396, "status": "ok", "timestamp": 1761449709871, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="843dc3c2"
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# %% [markdown]
# We load a subset of the M4 hourly dataset as a `pandas.DataFrame`
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 243} executionInfo={"elapsed": 883, "status": "ok", "timestamp": 1761452526676, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="4bcd73a8" outputId="56faf431-43ca-4b38-f47b-2fbe79f6fb37"
df = pd.read_csv(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv"
)
print(f"number of unique item_id: {df['item_id'].unique().shape[0]}")
print(f"number of samples: {df.shape[0]}")
df.head()

# %% [markdown] id="a64ddd43"
# AutoGluon expects time series data in [long format](https://doc.dataiku.com/dss/latest/time-series/data-formatting.html#long-format).
# Each row of the dataframe contains a single observation (timestep) of a single time series represented by
#
# - unique ID of the time series (`"item_id"`) as int or str
# - timestamp of the observation (`"timestamp"`) as a `pandas.Timestamp` or compatible format
# - numeric value of the time series (`"target"`)
#
# The raw dataset should always follow this format with at least three columns for unique ID, timestamp, and target value, but the names of these columns can be arbitrary.
# It is important, however, that we provide the names of the columns when constructing a `TimeSeriesDataFrame` that is used by AutoGluon.
# AutoGluon will raise an exception if the data doesn't match the expected format.

# %% colab={"base_uri": "https://localhost:8080/", "height": 238} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1761450892871, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="d73ddcc9" outputId="1f16a153-a56c-44be-ee5d-a962d418b4f1"
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
train_data.head()


# %% [markdown] id="bfee8b9b"
# We refer to each individual time series stored in a `TimeSeriesDataFrame` as an _item_.
# For example, items might correspond to different products in demand forecasting, or to different stocks in financial datasets.
# This setting is also referred to as a _panel_ of time series.
# Note that this is *not* the same as multivariate forecasting — AutoGluon generates forecasts for each time series individually, without modeling interactions between different items (time series).
#
# `TimeSeriesDataFrame` inherits from [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), so all attributes and methods of `pandas.DataFrame` are available in a `TimeSeriesDataFrame`.
# It also provides other utility functions, such as loaders for different data formats (see [TimeSeriesDataFrame](../../api/autogluon.timeseries.TimeSeriesDataFrame) for details).
#
# ## Training time series models with `TimeSeriesPredictor.fit`
# To forecast future values of the time series, we need to create a `TimeSeriesPredictor` object.
#
# Models in `autogluon.timeseries` forecast time series _multiple steps_ into the future.
# We choose the number of these steps — the _prediction length_ (also known as the _forecast horizon_) —  depending on our task.
# For example, our dataset contains time series measured at hourly _frequency_, so we set `prediction_length = 48` to train models that forecast up to 48 hours into the future.
#
# We instruct AutoGluon to save trained models in the folder `./autogluon-m4-hourly`.
# We also specify that AutoGluon should rank models according to [mean absolute scaled error (MASE)](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error), and that data that we want to forecast is stored in the column `"target"` of the `TimeSeriesDataFrame`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["634acfb9530b41b98e22a1bf39e491da", "a7f585e54f45406781896429bf659cfd", "c176d86b41574cd6a9d3f192015815fa", "93d740b1eb9a411e8b37fe57adceb436", "cf9e9356adb240519fdceff35d82feb4", "335a36d83e674f52920e17b8a7ced86d", "a9c6a860af544bdca1615330e69910ad", "dd01b263b8144bb08baef3f2fff4631d", "cff16fee17b64a97a7ef42c20248c8d0", "6cd87769b7044c2f84b7513c8184d09f", "31580cc9df6d43b2b9dde5c64221a679", "1a57ab01a9e74c8ab0394653246fcb85", "fd97313b1d2d4f7b93465c88eb757eb7", "83587f25e8d24f7b97e2158f4f5eb4b8", "c70a9e40b23c4ec1ad8dbe2ab9776b12", "b97621d64bb04941bd4d0dec942df058", "44cb7f307b674b58a230dc2acb7cc2e2", "8d1447a69ae04073adfda474d2055527", "03993346229d4a14a0784d26c8170fcf", "1e658bcd67254a03b2160cb43b9d3625", "69115c89b01e4e9c8ee498d6827aa1a4", "e6c74f0a713e4ef48dcf8993a36ae410"]} executionInfo={"elapsed": 399658, "status": "ok", "timestamp": 1761451315249, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="f7ef668c" outputId="292168d3-d134-407d-db51-33b2df756379"
predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)


# %% [markdown] id="91b3ddd4"
# Here we used the `"medium_quality"` presets and limited the training time to 10 minutes (600 seconds).
# The presets define which models AutoGluon will try to fit.
# For `medium_quality` presets, these are
# simple baselines (`Naive`, `SeasonalNaive`),
# statistical models (`ETS`, `Theta`),
# tree-based models based on LightGBM (`RecursiveTabular`, `DirectTabular`),
# a deep learning model `TemporalFusionTransformer`,
# and a weighted ensemble combining these.
# Other available presets for `TimeSeriesPredictor` are `"fast_training"`, `"high_quality"` and `"best_quality"`.
# Higher quality presets will usually produce more accurate forecasts but take longer to train.
#
# Inside `fit()`, AutoGluon will train as many models as possible within the given time limit.
# Trained models are then ranked based on their performance on an internal validation set.
# By default, this validation set is constructed by holding out the last `prediction_length` timesteps of each time series in `train_data`.
#
#
# ## Generating forecasts with `TimeSeriesPredictor.predict`
#
# We can now use the fitted `TimeSeriesPredictor` to forecast the future time series values.
# By default, AutoGluon will make forecasts using the model that had the best score on the internal validation set.
# The forecast always includes predictions for the next `prediction_length` timesteps, starting from the end of each time series in `train_data`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} executionInfo={"elapsed": 17027, "status": "ok", "timestamp": 1761451353062, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="4a238183" outputId="f20cfd02-4dce-4cf6-a0c8-83b1399cab2d"
predictions = predictor.predict(train_data)
predictions.head()


# %% [markdown] id="bfbca161"
# AutoGluon produces a _probabilistic_ forecast: in addition to predicting the mean (expected value) of the time series in the future, models also provide the quantiles of the forecast distribution.
# The quantile forecasts give us an idea about the range of possible outcomes.
# For example, if the `"0.1"` quantile is equal to `500.0`, it means that the model predicts a 10% chance that the target value will be below `500.0`.
#
# We will now visualize the forecast and the actually observed values for one of the time series in the dataset.
# We plot the mean forecast, as well as the 10% and 90% quantiles to show the range of potential outcomes.

# %% colab={"base_uri": "https://localhost:8080/", "height": 378} executionInfo={"elapsed": 2185, "status": "ok", "timestamp": 1761451669722, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="d2455126" outputId="b7b83a9f-9eba-4ed4-d639-b97bc4d5ffc7"
import matplotlib.pyplot as plt

# TimeSeriesDataFrame can also be loaded directly from a file
test_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv"
)

# Plot 4 randomly chosen time series and the respective forecasts
predictor.plot(
    test_data,
    predictions,
    quantile_levels=[0.1, 0.9],
    max_history_length=200,
    max_num_item_ids=4,
)

# %% [markdown] id="bc2d08f7"
# ## Evaluating the performance of different models
#
# We can view the performance of each model AutoGluon has trained via the `leaderboard()` method.
# We provide the test data set to the leaderboard function to see how well our fitted models are doing on the unseen test data.
# The leaderboard also includes the validation scores computed on the internal validation dataset.
#
# Note the test data includes both the forecast horizon (last `prediction_length` values of each time series) as well as the historical data (all except the last `prediction_last` values).
#
# In AutoGluon leaderboards, higher scores always correspond to better predictive performance.
# Therefore our MASE scores are multiplied by `-1`, such that higher "negative MASE"s correspond to more accurate forecasts.

# %% colab={"base_uri": "https://localhost:8080/", "height": 339} executionInfo={"elapsed": 11055, "status": "ok", "timestamp": 1761451750606, "user": {"displayName": "Richard Cui", "userId": "13653288654195820505"}, "user_tz": 300} id="2f4f8e9c" outputId="9021233b-13ab-48c3-e884-ffec21338753"
# The test score is computed using the last
# prediction_length=48 timesteps of each time series in test_data
predictor.leaderboard(test_data)

# %% [markdown] id="bd2fdfac"
# ## Summary
# We used `autogluon.timeseries` to make probabilistic multi-step forecasts on the M4 Hourly dataset.
# Check out [Forecasting Time Series - In Depth](forecasting-indepth.ipynb) to learn about the advanced capabilities of AutoGluon for time series forecasting.
