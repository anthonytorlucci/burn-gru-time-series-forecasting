use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::optim::AdamConfig;
use burn_gru_time_series_forecasting::gru_model::StackedGruRnnConfig;
use burn_gru_time_series_forecasting::time_series_training::StackedGruRnnTrainingConfig;
use std::path::Path;

pub fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    // ========== Model Training ==========
    let model_config = StackedGruRnnConfig::new(
        4,    // input_size: number of features (open, high, low, close)
        16,   // hidden_size
        4,    // output_size number of fetures (open, high, low, close)
        true, // bias
        0.1,  // dropout
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
