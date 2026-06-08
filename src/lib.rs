//! GRU-based time-series forecasting built on the [Burn](https://burn.dev) deep-learning framework.
//!
//! This crate provides everything needed to train and run a stacked Gated Recurrent Unit (GRU)
//! model on multivariate time-series data loaded from CSV files. It is structured around three
//! concerns:
//!
//! - **Model definitions** ([`gru_model`]) — [`CustomGruRnn`](gru_model::CustomGruRnn) (single
//!   layer) and [`StackedGruRnn`](gru_model::StackedGruRnn) (two layers), each exposing a
//!   sequence-to-sequence `forward` method and a sequence-to-one `forward_last` method.
//! - **Dataset and batching** ([`time_series_dataset`]) — sliding-window dataset over a CSV of
//!   OHLC stock prices, plus a batcher that converts [`StockTimeSeriesItemSample`](time_series_dataset::StockTimeSeriesItemSample)
//!   values into rank-3 input tensors and rank-2 target tensors.
//! - **Training loop** ([`time_series_training`]) — a single [`train`](time_series_training::train)
//!   function that builds dataloaders, configures a Burn [`Learner`](burn::train::Learner), runs
//!   supervised training, and saves the trained model.
//!
//! # When to use this crate
//!
//! Use this crate when you need a self-contained starting point for recurrent sequence modeling
//! with Burn. The dataset layer is intentionally coupled to a specific CSV schema (Date, Open,
//! High, Low, Close, Volume, Name); adapt [`time_series_dataset`] when working with other data
//! sources.
//!
//! # Quick start
//!
//! ```no_run
//! use burn::backend::Autodiff;
//! use burn::backend::wgpu::{Wgpu, WgpuDevice};
//! use burn::nn::Initializer;
//! use burn::optim::AdamConfig;
//! use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
//! use burn_gru_time_series_forecasting::time_series_training::{
//!     StackedGruRnnTrainingConfig, train,
//! };
//! use std::path::Path;
//!
//! type Backend = Wgpu<f32, i32>;
//!
//! let device = WgpuDevice::default();
//!
//! let model_config = StackedGruRnnConfig::new(
//!     4,    // input_size
//!     16,   // hidden_size
//!     4,    // output_size
//!     true, // bias
//!     0.1,  // dropout
//!     Initializer::XavierNormal { gain: 1.0 },
//! );
//!
//! let training_config = StackedGruRnnTrainingConfig::new(
//!     model_config,
//!     AdamConfig::new(),
//!     (0.7, 0.2, 0.1),
//! )
//! .with_num_epochs(25)
//! .with_batch_size(5)
//! .with_num_workers(1)
//! .with_seed(42)
//! .with_learning_rate(1e-4);
//!
//! train::<Autodiff<Backend>>(
//!     Path::new("data/GOOGL_2006-01-01_to_2018-01-01.csv"),
//!     Path::new("models/stock-time-series-gru"),
//!     training_config,
//!     &device,
//! );
//! ```

/// GRU model definitions and configuration types for time-series forecasting.
pub mod gru_model;

/// Sliding-window dataset, raw item types, and batcher for stock time-series CSV data.
pub mod time_series_dataset;

/// Training loop, `TrainStep`/`InferenceStep` implementations, and training configuration.
pub mod time_series_training;
