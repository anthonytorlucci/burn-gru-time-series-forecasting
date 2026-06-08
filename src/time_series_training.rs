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
    pub fn forecast(
        &self,
        source_tensor: Tensor<B, 3, Float>,
        target_tensor: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let output_tensor: Tensor<B, 2, Float> = self.forward_last(source_tensor);
        //--let output_flat: Tensor<B, 2, Float> = output.flatten(1, 2);
        //--let target_flat: Tensor<B, 2, Float> = target_tensor.flatten(1, 2);
        // MSE loss - see Gemeni response
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

    fn step(&self, batch: Self::Input) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forecast(batch.0, batch.1);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for StackedGruRnn<B> {
    type Input = (Tensor<B, 3, Float>, Tensor<B, 2, Float>);
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> RegressionOutput<B> {
        self.forecast(batch.0, batch.1)
    }
}

#[derive(Config, Debug)]
pub struct StackedGruRnnTrainingConfig {
    #[config(default = 2)]
    pub seq_length: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 12)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    pub model: StackedGruRnnConfig,
    pub optimizer: AdamConfig,
    pub train_test_valid_split: (f32, f32, f32),
}

// Create the directory to save the model and model config
fn create_artifact_dir(artifact_dir: &Path) {
    // Remove existing artifacts
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    csv_file_path: &Path,
    artifact_dir: &Path,
    config: StackedGruRnnTrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);

    // Save training config
    config
        .save(artifact_dir.join("config.json"))
        .expect("Config should be saved successfully");
    B::seed(device, config.seed);

    // Create the batcher.
    let batcher_train = StockTimeSeriesBatcher::<B>::new();
    let batcher_valid = StockTimeSeriesBatcher::<B::InnerBackend>::new();

    // Create the datasets
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

    // Create the dataloaders.
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

    // In Burn 0.21 the model, optimizer and learning rate scheduler are bundled into a
    // `Learner`, which is then handed to a `SupervisedTraining` runner that owns the
    // dataloaders, metrics and checkpointing configuration.
    let learner = Learner::new(
        config.model.init::<B>(device),                 // Initialize the model
        config.optimizer.init::<B, StackedGruRnn<B>>(), // Initialize the optimizer
        config.learning_rate,                           // f64 acts as a constant LR scheduler
    );

    let result = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        //.with_training_strategy(...) // defaults to single device based on the model
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    // `launch` returns a `LearningResult` whose `model` is the trained inference model.
    let model_trained = result.model;

    // Save the trained model
    model_trained
        .save_file(artifact_dir.join("StockTimeSeries"), &CompactRecorder::new())
        .expect("Trained StackedGruRnn model should be saved successfully");
}
