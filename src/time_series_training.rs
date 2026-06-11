//! Training pipeline for [`StackedGruRnn`] on windowed stock time-series data.
//!
//! This module wires together the model, dataset, and Burn training infrastructure.
//! It provides:
//!
//! - A [`forecast`](StackedGruRnn::forecast) method on [`StackedGruRnn`] that computes
//!   MSE loss and wraps the result in [`RegressionOutput`].
//! - [`TrainStep`] and [`InferenceStep`] implementations so the model integrates with
//!   Burn's [`Learner`] / [`SupervisedTraining`] API without additional glue code.
//! - [`StackedGruRnnTrainingConfig`] — a serialisable configuration struct that bundles
//!   all hyperparameters needed to reproduce a training run.
//! - [`train`] — a top-level function that builds the dataloaders, initialises the
//!   learner, runs training, and saves the final model and config to disk.
//!
//! # When to use this module
//!
//! Use this module when you want to train a [`StackedGruRnn`] model from a CSV file
//! that follows the same schema as [`WindowedStockTimeSeriesDataset`]. For inference
//! only (no gradient computation), load the saved model directly and call
//! [`StackedGruRnn::forward_last`].
//!
//! # Examples
//!
//! ```no_run
//! use std::path::Path;
//! use burn::backend::{Autodiff, NdArray};
//! use burn::optim::AdamConfig;
//! use burn::nn::Initializer;
//! use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
//! use burn_gru_time_series_forecasting::time_series_training::{
//!     StackedGruRnnTrainingConfig, train,
//! };
//!
//! type MyBackend = Autodiff<NdArray>;
//!
//! let config = StackedGruRnnTrainingConfig::new(
//!     StackedGruRnnConfig::new(
//!         4,
//!         32,
//!         4,
//!         true,
//!         0.1,
//!         Initializer::XavierNormal { gain: 1.0 },
//!     ),
//!     AdamConfig::new(),
//!     (0.7, 0.15, 0.15),
//! );
//!
//! train::<MyBackend>(
//!     Path::new("data/stocks.csv"),
//!     Path::new("artifacts/"),
//!     config,
//!     &Default::default(),
//! );
//! ```

use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::prelude::Config;
use burn::record::CompactRecorder;
use burn::tensor::Float;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::InferenceStep;
use burn::train::Learner;
use burn::train::RegressionOutput;
use burn::train::SupervisedTraining;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::metric::LossMetric;
use std::path::Path;

use crate::gru_model::{StackedGruRnn, StackedGruRnnConfig};
use crate::time_series_dataset::{StockTimeSeriesBatcher, WindowedStockTimeSeriesDataset};

impl<B: Backend> StackedGruRnn<B> {
    /// Computes a forward pass and returns loss, predictions, and targets as [`RegressionOutput`].
    ///
    /// Runs `source_tensor` through [`StackedGruRnn::forward_last`] to obtain a prediction
    /// of shape `[batch_size, output_size]`, then computes mean-squared-error loss against
    /// `target_tensor` using [`Reduction::Auto`].
    ///
    /// `source_tensor` must have shape `[batch_size, sequence_length, input_size]` and
    /// `target_tensor` must have shape `[batch_size, output_size]`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use burn::backend::NdArray;
    /// # use burn::nn::Initializer;
    /// # use burn::tensor::Tensor;
    /// # use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
    /// # let device = Default::default();
    /// # let model = StackedGruRnnConfig::new(4, 16, 4, true, 0.0,
    /// #     Initializer::XavierNormal { gain: 1.0 }).init::<NdArray>(&device);
    /// let source: Tensor<NdArray, 3> = Tensor::zeros([2, 10, 4], &device);
    /// let target: Tensor<NdArray, 2> = Tensor::zeros([2, 4], &device);
    /// let output = model.forecast(source, target);
    /// ```
    pub fn forecast(
        &self,
        source_tensor: Tensor<B, 3, Float>,
        target_tensor: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let output_tensor: Tensor<B, 2, Float> = self.forward_last(source_tensor);
        let loss = MseLoss::new().forward(
            output_tensor.clone(),
            target_tensor.clone(),
            Reduction::Auto,
        );
        RegressionOutput::new(loss, output_tensor, target_tensor)
    }
}

impl<B: AutodiffBackend> TrainStep for StackedGruRnn<B> {
    type Input = (Tensor<B, 3, Float>, Tensor<B, 2, Float>);
    type Output = RegressionOutput<B>;

    /// Runs a training step by computing loss and backpropagating gradients.
    ///
    /// Delegates to [`StackedGruRnn::forecast`] and calls `.backward()` on the returned
    /// loss before wrapping the result in [`TrainOutput`].
    fn step(&self, batch: Self::Input) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forecast(batch.0, batch.1);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for StackedGruRnn<B> {
    type Input = (Tensor<B, 3, Float>, Tensor<B, 2, Float>);
    type Output = RegressionOutput<B>;

    /// Runs an inference step without gradient computation.
    ///
    /// Delegates directly to [`StackedGruRnn::forecast`]. No gradients are computed
    /// because the backend `B` is not required to implement [`AutodiffBackend`].
    fn step(&self, batch: Self::Input) -> RegressionOutput<B> {
        self.forecast(batch.0, batch.1)
    }
}

/// Serialisable hyperparameter bundle for a [`StackedGruRnn`] training run.
///
/// All fields have sensible defaults derived via the [`Config`] macro. Override
/// individual fields with the generated `with_*` builder methods before calling
/// [`train`].
///
/// # Examples
///
/// ```no_run
/// use burn::optim::AdamConfig;
/// use burn::nn::Initializer;
/// use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
/// use burn_gru_time_series_forecasting::time_series_training::StackedGruRnnTrainingConfig;
///
/// let config = StackedGruRnnTrainingConfig::new(
///     StackedGruRnnConfig::new(4, 32, 4, true, 0.1, Initializer::XavierNormal { gain: 1.0 }),
///     AdamConfig::new(),
///     (0.7, 0.15, 0.15),
/// )
/// .with_num_epochs(20)
/// .with_learning_rate(1e-3);
/// ```
#[derive(Config, Debug)]
pub struct StackedGruRnnTrainingConfig {
    /// Number of time steps in each input window fed to the model.
    #[config(default = 2)]
    pub seq_length: usize,
    /// Number of complete passes over the training dataset.
    #[config(default = 10)]
    pub num_epochs: usize,
    /// Number of samples per mini-batch during training and validation.
    #[config(default = 12)]
    pub batch_size: usize,
    /// Number of parallel worker threads used by the dataloader.
    #[config(default = 4)]
    pub num_workers: usize,
    /// Random seed for reproducible weight initialisation and dataloader shuffling.
    #[config(default = 42)]
    pub seed: u64,
    /// Constant learning rate passed to the Adam optimiser.
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    /// Architecture configuration for the [`StackedGruRnn`] model.
    pub model: StackedGruRnnConfig,
    /// Optimiser configuration; defaults are applied via [`AdamConfig`].
    pub optimizer: AdamConfig,
    /// Proportions `(train, test, valid)` used to split the source dataset.
    ///
    /// Values need not sum to exactly `1.0`; they are normalised automatically.
    pub train_test_valid_split: (f32, f32, f32),
}

/// Removes and recreates the artifact directory at `artifact_dir`.
///
/// Any files previously written to that path are deleted before the directory
/// is created so that stale checkpoints and configs do not persist across runs.
fn create_artifact_dir(artifact_dir: &Path) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Trains a [`StackedGruRnn`] model and saves the result to `artifact_dir`.
///
/// Reads time-series data from the CSV file at `csv_file_path`, constructs
/// training and validation [`WindowedStockTimeSeriesDataset`]s according to the
/// split ratios in `config`, and runs supervised training for the configured
/// number of epochs.
///
/// On completion the function writes two files to `artifact_dir`:
///
/// - `config.json` — the full [`StackedGruRnnTrainingConfig`] used for this run.
/// - `StockTimeSeries.mpk.gz` — the trained model weights in compact format.
///
/// `artifact_dir` is always recreated from scratch; any prior contents are deleted.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use burn::backend::{Autodiff, NdArray};
/// use burn::optim::AdamConfig;
/// use burn::nn::Initializer;
/// use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
/// use burn_gru_time_series_forecasting::time_series_training::{
///     StackedGruRnnTrainingConfig, train,
/// };
///
/// type MyBackend = Autodiff<NdArray>;
///
/// let config = StackedGruRnnTrainingConfig::new(
///     StackedGruRnnConfig::new(4, 32, 4, true, 0.1, Initializer::XavierNormal { gain: 1.0 }),
///     AdamConfig::new(),
///     (0.7, 0.15, 0.15),
/// );
///
/// train::<MyBackend>(
///     Path::new("data/stocks.csv"),
///     Path::new("artifacts/"),
///     config,
///     &Default::default(),
/// );
/// ```
///
/// # Panics
///
/// Panics if `config.json` cannot be written to `artifact_dir`, if either the
/// training or validation dataset cannot be built from `csv_file_path`, or if
/// the trained model cannot be saved.
pub fn train<B: AutodiffBackend>(
    csv_file_path: &Path,
    artifact_dir: &Path,
    config: StackedGruRnnTrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);

    config
        .save(artifact_dir.join("config.json"))
        .expect("Config should be saved successfully");
    B::seed(device, config.seed);

    let batcher_train = StockTimeSeriesBatcher::<B>::new();
    let batcher_valid = StockTimeSeriesBatcher::<B::InnerBackend>::new();

    let train_dataset: WindowedStockTimeSeriesDataset = WindowedStockTimeSeriesDataset::new(
        &csv_file_path.to_path_buf(),
        "train",
        config.train_test_valid_split,
        config.seq_length,
    )
    .expect("Failed to build training dataset");
    let valid_dataset: WindowedStockTimeSeriesDataset = WindowedStockTimeSeriesDataset::new(
        &csv_file_path.to_path_buf(),
        "valid",
        config.train_test_valid_split,
        config.seq_length,
    )
    .expect("Failed to build validation dataset");

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let learner = Learner::new(
        config.model.init::<B>(device),
        config.optimizer.init::<B, StackedGruRnn<B>>(),
        config.learning_rate,
    );

    let result = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    let model_trained = result.model;

    model_trained
        .save_file(artifact_dir.join("StockTimeSeries"), &CompactRecorder::new())
        .expect("Trained StackedGruRnn model should be saved successfully");
}
