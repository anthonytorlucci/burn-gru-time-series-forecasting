//! Training binary for GRU-based stock price forecasting.
//!
//! Reads OHLC (open, high, low, close) price data from a CSV file, trains a
//! two-layer stacked GRU model using the Burn framework on the WGPU backend,
//! and saves the trained model artefacts to disk.
//!
//! # Usage
//!
//! ```text
//! cargo run --release
//! ```
//!
//! The binary expects the following paths relative to the working directory:
//!
//! - `data/GOOGL-stock-time-series/GOOGL_2006-01-01_to_2018-01-01.csv` — input CSV
//! - `models/stock-time-series-gru/` — output directory for the saved model and config
//!
//! # Output
//!
//! After training completes the output directory contains:
//!
//! - `config.json` — serialised [`StackedGruRnnTrainingConfig`]
//! - `StockTimeSeries.mpk.gz` — trained model weights in compact record format
//! - Burn training metrics and checkpoints written by the [`SupervisedTraining`] runner

use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::optim::AdamConfig;
use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
use burn_gru_time_series_forecasting::time_series_training::StackedGruRnnTrainingConfig;
use std::path::Path;

/// Configures and launches model training on the default WGPU device.
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let model_config = StackedGruRnnConfig::new(
        4,
        16,
        4,
        true,
        0.1,
        Initializer::XavierNormal { gain: (1.0) },
    );
    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8);
    let training_config =
        StackedGruRnnTrainingConfig::new(model_config, optimizer_config, (0.7, 0.2, 0.1))
            .with_num_epochs(25)
            .with_batch_size(5)
            .with_num_workers(1)
            .with_seed(42)
            .with_learning_rate(1e-4);

    burn_gru_time_series_forecasting::time_series_training::train::<Autodiff<MyBackend>>(
        &Path::new("data")
            .join("GOOGL-stock-time-series")
            .join("GOOGL_2006-01-01_to_2018-01-01.csv"),
        &Path::new("models").join("stock-time-series-gru"),
        training_config,
        &device,
    );
}
