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

# %% [markdown] id="3BtsV15yWwQC"
# # Forecasting with Chronos-2
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/stable/docs/tutorials/timeseries/forecasting-chronos.ipynb)
# [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/stable/docs/tutorials/timeseries/forecasting-chronos.ipynb)
#
# AutoGluon-TimeSeries (AG-TS) includes the [Chronos](https://github.com/amazon-science/chronos-forecasting) family of forecasting models. Chronos models are pretrained on a large collection of real and synthetic time series data, enabling accurate out-of-the-box forecasts on new data.
#
# AG-TS provides a robust and user-friendly way to work with Chronos through the familiar `TimeSeriesPredictor` API. It allows users to backtest models, compare them with other forecasting approaches, and ensemble Chronos with other models to build robust forecasting pipelines. This tutorial demonstrates how to:
#
# - Use Chronos-2 in **zero-shot** mode to generate forecasts without dataset-specific training
# - **Fine-tune** Chronos-2 on custom data to improve accuracy
#
# :::{note}
#
# **New in v1.5:** AutoGluon now features [Chronos-2](https://arxiv.org/abs/2510.15821) — the latest version of Chronos models with _zero-shot_ support for covariates and a [90%+ win-rate](https://huggingface.co/spaces/autogluon/fev-bench) over Chronos-Bolt. The older version of this tutorial with the Chronos-Bolt model is available [here](https://auto.gluon.ai/1.4.0/tutorials/timeseries/forecasting-chronos.html).
#
# :::

# %% [raw]
# # We use uv for faster installation
# !pip install uv
# !uv pip install -q autogluon.timeseries --system
# !uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

# %%
# use only one GPU if there are multiple GPUs available, to avoid OOM issues
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# %% [markdown] id="xV7UtCFgWwQE"
# ## Getting started with Chronos-2
#
# Being a pretrained model for zero-shot forecasting, Chronos is different from other models available in AG-TS.
# Specifically, by default, Chronos models do not really `fit` time series data. However, when `predict` is called, they perform _zero-shot inference_ by using the provided contextual information. In this aspect, they behave like local statistical models such as ETS or ARIMA, where all computation happens during inference.
#
# AutoGluon supports the original Chronos models (e.g., [`chronos-t5-large`](https://huggingface.co/autogluon/chronos-t5-large)), the Chronos-Bolt models (e.g., [`chronos-bolt-base`](https://huggingface.co/autogluon/chronos-bolt-base)), and the latest Chronos-2 models (e.g., [`chronos-2`](https://huggingface.co/autogluon/chronos-2)). The following table compares the capabilities of the three model families.
#
# | Capability | Chronos | Chronos-Bolt | Chronos-2 |
# |------------|---------|--------------|-----------|
# | Univariate Forecasting | ✅ | ✅ | ✅ |
# | Cross-learning across items | ❌ | ❌ | ✅ |
# | Multivariate Forecasting | ❌ | ❌ | ✅ |
# | Past-only (real/categorical) covariates | ❌ | ❌ | ✅ |
# | Known future (real/categorical) covariates | 🧩 | 🧩 | ✅ |
# | Fine-tuning support | ✅ | ✅ | ✅ |
# | Max. Context Length | 512 | 2048 | 8192 |
# | Max. Prediction Length | 64 | 64 | 1024 |
#
#
# The easiest way to get started with Chronos is through the model-specific presets.
#
# - **(recommended)** The Chronos-2 models can be accessed using the `"chronos2_small"` and `"chronos2"` presets.
# - The Chronos-Bolt️ models can be accessed using the `"bolt_tiny"`, `"bolt_mini"`, `"bolt_small"` and `"bolt_base"` presets.
#
# Alternatively, Chronos models can be combined with other time series models using presets `"medium_quality"`, `"high_quality"` and `"best_quality"`. More details about these presets are available in the documentation for [`TimeSeriesPredictor.fit`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html).
#
#
# 🧩 Chronos/Chronos-Bolt do not natively support future covariates, but they can be combined with external covariate regressors. This only models per-timestep effects, not effects across time. In contrast, Chronos-2 supports all covariate types natively.

# %% [markdown] id="vzCDJDlEWwQF"
# ## Zero-shot forecasting
#
# ### Univariate Forecasting

# %% [markdown] id="jXEr8FdgWwQF"
# Let's work with a subset of the [Australian Electricity Demand dataset](https://zenodo.org/records/4659727) to see Chronos-2 in action.
#
# First, we load the dataset as a [TimeSeriesDataFrame](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html).

# %% id="zr6JjNneWwQG"
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# %% id="E6Z0NGjBWwQG" outputId="4bc73bfb-218d-4b75-f7cb-3a089fc88465"
data_csv_url = "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
data = TimeSeriesDataFrame.from_path(data_csv_url)
# data_df = pd.read_csv(data_csv_url)
# print(data_df.head())
# data = TimeSeriesDataFrame.from_data_frame(data_df)
print("data shape", data.shape)
print(data.head())
print("number of unique time series:", data.index.get_level_values("item_id").nunique())

# %% [markdown] id="nG2bniu5WwQH"
# Next, we create the [TimeSeriesPredictor](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html) and select the `"chronos2"` presets to use the Chronos-2 (120M) model in zero-shot mode.

# %% id="wAyToWF4WwQI" outputId="f8745321-4fc7-4162-fa65-2dfe9cdac687" tags=["hide-output"]
num_test_windows = 3
prediction_length = 48
train_data, test_data = data.train_test_split(num_test_windows * prediction_length)
print("shape of train data: ", train_data.shape)
print(
    "shape of test data (train length + num_test_windows * prediction_length"
    " * number of time series): ",
    test_data.shape,
)

# %%
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data,
    presets="chronos2",
    # presets="best_quality",
)

# %%
predictor.leaderboard(test_data)

# %% [markdown] id="AsDJrs4mWwQI"
# As promised, Chronos does not take any time to `fit`. The `fit` call merely serves as a proxy for the `TimeSeriesPredictor` to do some of its chores under the hood, such as inferring the frequency of time series and saving the predictor's state to disk.
#
# Let's use the `predict` method to generate forecasts.

# %% id="ZJveK28-WwQJ" outputId="4dc9cd36-324b-4496-dcf5-736163ed0caf"
predictions = predictor.predict(train_data)
print("predictions shape: ", predictions.shape)
predictions.head()

# %%
import matplotlib.pyplot as plt

# Plot predictions for all the time series
item_ids = test_data.item_ids.tolist()
predictor.plot(test_data, predictions, max_history_length=300, item_ids=item_ids)
plt.show()

# %% [markdown] id="CcUP7k-tWwQJ"
# We get a dataframe with the point forecast (`mean`) and nine quantiles which capture the uncertainty in the forecasts. Custom quantile levels can be specified as follows:
# ```py
# TimeSeriesPredictor(..., quantile_levels=[0.05, 0.1, 0.5, 0.9, 0.95])
# ```

# %% [markdown] id="QiHhY0VXWwQJ"
# AG-TS also makes it easy to generate predictions for multiple backtest dates and to visualize the models' predictions.

# %% id="rZ_W_VJoWwQJ" outputId="e4da56d9-84e3-4241-ef59-f10c039451ff"
# Generate predictions for multiple windows
predictions_per_window = predictor.backtest_predictions(
    test_data, num_val_windows=num_test_windows
)

# Plot predictions for all time series
item_ids = test_data.item_ids.tolist()
all_predictions = pd.concat(predictions_per_window)
predictor.plot(test_data, all_predictions, max_history_length=300, item_ids=item_ids)

# Optional: Plot the cutoff dates with dashed vertical lines
for cutoff in range(-num_test_windows * prediction_length, 0, prediction_length):
    for i, (ax, item_id) in enumerate(zip(plt.gcf().axes, item_ids)):
        cutoff_timestamp = test_data.loc[item_id].index[cutoff]
        ax.axvline(cutoff_timestamp, color="gray", linestyle="--")
plt.show()

# %% [markdown] id="IjZYv-fcWwQK"
# ## Forecasting with covariates

# %% [markdown] id="6AKWOyekWwQK"
# The previous example showed Chronos-2 in action on a univariate forecasting task, i.e., only the historical data of the target time series for making predictions. However, in real-world scenarios, additional exogenous information related to the target series (e.g., weather forecasts, holidays, promotions) is often available. These exogenous time series, often referred to as covariates, may either be observed only in the past (past-only) or also in the forecast horizon (known future). Leveraging this information when making predictions can improve forecast accuracy.
#
# Chronos-2 natively supports (dynamic) covariates, past-only and known-future, real-valued or categorical. Let's see how we can use Chronos-2 to forecast with covariates on a **Electrical Load Forecasting** task.

# %% id="0nVdNe_1WwQK" outputId="3544d25c-69f1-4175-803e-3e12687ac1b0"
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/bull/test.parquet",
    id_column="id",
)
print(data.head())
print("number of unique time series:", data.index.get_level_values("item_id").nunique())

# %% [markdown] id="ASueiy-ZWwQK"
# The goal is to forecast next day's (24 hours) load using historical load and known weather covariates: air temperature, dew temperature and sea level pressure. Since future weather information is not known in advance, weather forecasts are typically used as known covariates.

# %% id="YBP7frWNWwQL" outputId="0ef5607d-aecd-4394-d5b2-4331e43d8c0d"
prediction_length = 24
train_data, test_data = data.train_test_split(prediction_length=prediction_length)
print("shape of train data: ", train_data.shape)
print("shape of test data: ", test_data.shape)

# %% [markdown] id="J1SF46hRWwQL"
# The following code uses Chronos-2 in the TimeSeriesPredictor to forecast the `load` for the next 24 hours. We use the _univariate_ [Chronos-Bolt (Small)](https://huggingface.co/autogluon/chronos-bolt-small) model as a baseline for comparison.
#
# Note that we have specified the target column we are interested in forecasting and the names of known covariates while constructing the TimeSeriesPredictor. Any other columns, if present, will be used as past-only covariates.

# %% id="MzXjTYmwWwQL" outputId="8cf9991d-e392-42d2-e3b9-485eec73f6d5" tags=["hide-output"]
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="load",
    known_covariates_names=["airtemperature", "dewtemperature", "sealvlpressure"],
    eval_metric="MASE",
).fit(
    train_data,
    hyperparameters={"Chronos": {}, "Chronos2": {}},
    enable_ensemble=False,
    time_limit=60,
)

# %% [markdown] id="X6ljCuD3WwQL"
# Once the predictor has been fit, we can evaluate it on the test dataset and generate the leaderboard. We see that Chronos-2, which utilizes covariates, produces a significantly more accurate forecast on the test set compared to Chronos-Bolt, which does not utilize covariates.
#
# Note that all AutoGluon-TimeSeries models report scores in a "higher is better" format, meaning that most forecasting error metrics like MASE are multiplied by -1 when reported.

# %% id="z7IMlTsmWwQM" outputId="5c57e534-3224-4142-b24e-aa0272daab0f"
predictor.leaderboard(test_data)

# %% [markdown] id="lxZgGhbYWwQM"
# We can also use the predictor to compute features importances to understand which exogenous features are affecting the prediction accuracy the most.

# %% id="31oHVeODWwQM" outputId="48b6d3d6-31e0-4f4a-8b7c-b3e2d9bb4a57"
predictor.feature_importance(test_data, model="Chronos2", relative_scores=True)

# %% [markdown] id="AXN2OqAzWwQM"
# With `relative_scores=True`, this method returns relative (percentage) improvements in the `eval_metric` due to each feature. In this example, the `airtemperature` feature is the most important for accurate forecasting, yielding a ~32% error reduction on the test set.
#
# Note that covariates may not always be useful and using more covariates does not necessarily imply more accurate forecasts. With Chronos-2, AutoGluon makes it easy for users to quickly validate different configurations and find ones that perform best on held-out data.

# %% [markdown] id="IS3UNz-SWwQM"
# ## Fine-tuning
#
# We have seen above how Chronos-2 models can produce forecasts in zero-shot mode, both with and without covariates. AutoGluon also makes it easy to fine-tune Chronos models on a specific dataset to maximize the predictive accuracy.
#
# The following snippet specifies two settings for the Chronos-2 model: zero-shot and fine-tuned. `TimeSeriesPredictor` will perform a lightweight fine-tuning of the pretrained model on the provided training data. We add name suffixes to easily identify the zero-shot and fine-tuned versions of the model.
#
# :::{note}
#
# If you are fine-tuning on a machine with multiple GPUs, we strongly recommend setting the `CUDA_VISIBLE_DEVICES` environment variable to ensure that only a single GPU is visible.
#
# :::

# %% id="lSHydpr6WwQN" outputId="47ba7c50-27cf-43b3-9daf-99f217ed746b" tags=["hide-output"]
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="load",
    known_covariates_names=["airtemperature", "dewtemperature", "sealvlpressure"],
    eval_metric="MASE",
).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos2": [
            # Zero-shot model
            {"ag_args": {"name_suffix": "ZeroShot"}},
            # Fine-tuned model
            {"fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=300,  # time limit in seconds
    enable_ensemble=False,
)

# %% [markdown] id="DsJLdMPtWwQN"
# Here we used the default fine-tuning configuration for Chronos-2 by only specifying `"fine_tune": True`. By default, Chronos-2 is fine-tuned with a low-rank adapter (LoRA) to reduce memory and disk footprint. AutoGluon makes it easy to change other parameters for fine-tuning such as the mode, number of steps or learning rate.
# ```python
# predictor.fit(
#     ...,
#     hyperparameters={"Chronos2": {"fine_tune": True, "fine_tune_mode": "full", "fine_tune_lr": 1e-4, "fine_tune_steps": 2000, "fine_tune_batch_size": 32}},
# )
# ```
#
# For the full list of fine-tuning options, see the Chronos-2 documentation in [Forecasting Model Zoo](forecasting-model-zoo.md#autogluon.timeseries.models.Chronos2Model).
#
#
# After fitting, we can evaluate the two model variants on the test data and generate a leaderboard.

# %% id="HADvkXW_WwQN" outputId="706fedc2-ecc7-4a8f-d5d9-dec0341aa3b3"
predictor.leaderboard(test_data)

# %% [markdown] id="k2rGGYhkWwQN"
# Fine-tuning resulted in a more accurate model, as shown by the better `score_test` on the test set.

# %% [markdown] id="q152cvQ7WwQN"
# ## FAQ
#
#
# #### How accurate is Chronos-2?
#
# Chronos-2 is the best performing (last updated: Dec 2025) time series foundation model across multiple benchmarks, including [fev-bench](https://huggingface.co/spaces/autogluon/fev-bench), [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [Chronos Bench II](https://arxiv.org/abs/2403.07815). Details empirical results can be found in the [Chronos-2 technical report](https://arxiv.org/abs/2510.15821). The accuracy of Chronos-2 often exceeds statistical baseline models and task-specific deep learning models such as `DeepAR` and `TemporalFusionTransformer`.
#
# #### Does fine-tuning always improve Chronos-2's forecasting accuracy?
#
# Fine-tuning a foundation model like Chronos-2 involves many hyperparameter choices. AG-TS provides reasonable defaults that performed well in large-scale benchmarking, but they may not be optimal for every use case. We recommend fine-tuning only when you have a reasonable number of time series and sufficient historical data (e.g., >100 time series with a median history length larger than `3 * prediction_length`), as limited data can lead to overfitting or degraded performance. If you observe degraded accuracy, we recommend increasing the size of the training data and experimenting with different fine-tuning hyperparameters.
#
# Alternatively, you can use an ensemble of zero-shot Chronos-2 and fine-tuned Chronos-2 (Small) to construct a robust predictor, available via the `chronos2_ensemble` preset:
#
# ```py
# predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
#     ...,
#     presets="chronos2_ensemble",
# )
# ```
#
# #### What is the recommended hardware for running Chronos models?
#
# We recommend using a machine with a GPU for best performance, especially for fine-tuning. For reference, we tested the models on AWS `g5.2xlarge` instances with NVIDIA A10G GPUs (24 GiB GPU memory) and 32 GiB of system memory. However, Chronos-2, Chronos-Bolt, and Chronos (up to small size) can also run on consumer GPUs and CPUs with reasonable inference times.
#
# #### Why do my predictions change with the `batch_size`?
#
# By default, AutoGluon enables Chronos-2’s cross_learning mode, where the model makes joint predictions across time series within a batch. This often improves accuracy but also makes results sensitive to the `batch_size`. You can disable this mode with:
#
# ```python
# predictor.fit(
#     ...,
#     hyperparameters={"Chronos2": {"cross_learning": False}},
# )
# ```
#
# #### Where can I ask specific questions on Chronos?
#
# Members of the AutoGluon team are among the core developers of Chronos. So you can ask Chronos-related questions on [AutoGluon's GitHub](https://github.com/autogluon/autogluon) or on [Chronos' GitHub](https://github.com/amazon-science/chronos-forecasting/discussions).

# %% [markdown] id="SPb8AXySWwQO"
#
