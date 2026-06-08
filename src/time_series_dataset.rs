// Recurrent Neural Networks (RNNs) are designed to process sequential data, where the order of information is crucial. Unlike traditional neural networks that treat data points as independent, RNNs use an internal memory or "hidden state" to consider the context of past inputs when processing new ones.
//
// Preparing data for an RNN
//
// To train an RNN, sequential data must be preprocessed into a numerical format with a specific three-dimensional structure.
//
// Scale numerical features to a uniform range (e.g., between 0 and 1) to help the model converge faster during training. For example, a z-score normalization (centering data around 0 with a standard deviation of 1) is a common method.
//
// Transform the numerical data into sequences using a "sliding window" or "look-back" approach. For a time series, this involves creating pairs of input sequences and target outputs. For example, to predict the next value in a series, you could use the last 10 steps as input.
//
// The data must be reshaped into a three-dimensional tensor with the following structure: (number of samples, sequence length, number of features).
// * Number of samples: The total number of sequences in your dataset.
// * Sequence length: The number of time steps in each sequence.
// * Number of features: The number of variables at each time step.

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use burn::tensor::{Float, Shape, Tensor, TensorData};
use log;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub struct DatasetError {
    details: String,
}

impl DatasetError {
    fn new(msg: &str) -> DatasetError {
        DatasetError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for DatasetError {
    fn description(&self) -> &str {
        &self.details
    }
}

impl From<io::Error> for DatasetError {
    fn from(error: io::Error) -> Self {
        DatasetError {
            details: error.to_string(),
        }
    }
}

impl From<csv::Error> for DatasetError {
    fn from(error: csv::Error) -> Self {
        DatasetError {
            details: error.to_string(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StockTimeSeriesItem {
    #[serde(rename = "Date")]
    pub date: String,
    #[serde(rename = "Open")]
    pub open: f32,
    #[serde(rename = "High")]
    pub high: f32,
    #[serde(rename = "Low")]
    pub low: f32,
    #[serde(rename = "Close")]
    pub close: f32,
    #[serde(rename = "Volume")]
    pub volume: String,
    #[serde(rename = "Name")]
    pub name: String,
}

// --- Define Input/Target sizes and the Sample Struct ---
// Let's use 'Open', 'High', 'Low', and 'Close' as our features.
const INPUT_SIZE: usize = 4;
// We'll predict the 'Open', 'High', 'Low', and 'Close' of the next day.
const TARGET_SIZE: usize = 4;

/// Represents one complete training sample: a sequence of inputs and a target.
#[derive(Clone, Debug)]
pub struct StockTimeSeriesItemSample {
    pub inputs: Vec<[f32; INPUT_SIZE]>,
    pub target: [f32; TARGET_SIZE],
}

/// This dataset wraps the raw data and provides sliding windows as samples.
pub struct WindowedStockTimeSeriesDataset {
    items: Vec<StockTimeSeriesItem>,
    sequence_length: usize,
}

impl WindowedStockTimeSeriesDataset {
    /// Creates a new windowed dataset.
    ///
    /// # Arguments
    ///
    /// * `file_path`: Path to the CSV file.
    /// * `split`: "train", "test", or "valid".
    /// * `split_size`: Tuple of (train, test, valid) proportions.
    /// * `sequence_length`: The hyperparameter for your input sequence length.
    pub fn new(
        file_path: &PathBuf,
        split: &str,
        split_size: (f32, f32, f32),
        sequence_length: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if !file_path.exists() {
            return Err(Box::new(DatasetError::new("File does not exist")));
        }

        let mut rdr = csv::ReaderBuilder::new().from_path(file_path)?;
        let mut items: Vec<StockTimeSeriesItem> = Vec::new();
        for result in rdr.deserialize() {
            let record: StockTimeSeriesItem = result?;
            items.push(record);
        }

        let total_items = items.len();
        if total_items == 0 {
            return Err(Box::new(DatasetError::new("Dataset is empty")));
        }

        let (mut train_prop, mut test_prop, valid_prop) = split_size;
        let sum = train_prop + test_prop + valid_prop;
        if sum <= 0.0 {
            return Err(Box::new(DatasetError::new(
                "Invalid split sizes: sum must be > 0",
            )));
        }
        if (sum - 1.0).abs() > f32::EPSILON {
            train_prop /= sum;
            test_prop /= sum;
            //valid_prop /= sum;
        }

        let train_count = (total_items as f32 * train_prop).floor() as usize;
        let test_count = (total_items as f32 * test_prop).floor() as usize;
        //let valid_count = total_items.saturating_sub(train_count + test_count); // Use remainder for validation

        let train_end = train_count.min(total_items);
        let test_end = (train_end + test_count).min(total_items);
        let valid_end = total_items; // Use all remaining data

        log::info!(
            "Dataset split ({}): Train [0..{}], Test [{}..{}], Valid [{}..{}]",
            split,
            train_end,
            train_end,
            test_end,
            test_end,
            valid_end
        );

        let split_items: Vec<StockTimeSeriesItem> = match split {
            "train" => items[0..train_end].to_vec(),
            "test" => items[train_end..test_end].to_vec(),
            "valid" | "validation" => items[test_end..valid_end].to_vec(),
            other => {
                return Err(Box::new(DatasetError::new(&format!(
                    "Invalid split name `{}`; expected \"train\", \"test\", or \"valid\"",
                    other
                ))));
            }
        };

        if split_items.is_empty() {
            log::warn!("Dataset split '{}' resulted in 0 items.", split);
        }

        if split_items.len() <= sequence_length {
            return Err(Box::new(DatasetError::new(&format!(
                "Dataset split '{}' has {} items, which is not greater than sequence length {}",
                split,
                split_items.len(),
                sequence_length
            ))));
        }

        Ok(Self {
            items: split_items,
            sequence_length,
        })
    }

    /// Helper to extract features from a raw data item.
    fn item_to_features(item: &StockTimeSeriesItem) -> [f32; INPUT_SIZE] {
        [item.open, item.high, item.low, item.close]
    }

    /// Helper to extract target features from a raw data item.
    fn item_to_target(item: &StockTimeSeriesItem) -> [f32; TARGET_SIZE] {
        [item.open, item.high, item.low, item.close]
    }
}

/// Implement `Dataset` for our new `WindowedStockDataset`.
/// The item type `I` is now `StockTimeSeriesItemSample`.
impl Dataset<StockTimeSeriesItemSample> for WindowedStockTimeSeriesDataset {
    fn get(&self, index: usize) -> Option<StockTimeSeriesItemSample> {
        // Check if we have enough items for a full sequence *plus* one target item
        if index + self.sequence_length >= self.items.len() {
            return None;
        }

        // Extract the input sequence
        let inputs: Vec<[f32; INPUT_SIZE]> = (index..index + self.sequence_length)
            .map(|i| Self::item_to_features(&self.items[i]))
            .collect();

        // Extract the target (the item immediately after the sequence)
        let target = Self::item_to_target(&self.items[index + self.sequence_length]);

        Some(StockTimeSeriesItemSample { inputs, target })
    }

    /// The number of *possible sequences* we can create.
    fn len(&self) -> usize {
        // We can't create a sequence from the last `sequence_length` items
        // because they don't have a subsequent target.
        self.items.len().saturating_sub(self.sequence_length)
    }
}

/// This Batcher knows how to handle `StockTimeSeriesItemSample`.
pub struct StockTimeSeriesBatcher<B: Backend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for StockTimeSeriesBatcher<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> StockTimeSeriesBatcher<B> {
    pub fn new() -> Self {
        Self {
            _backend: std::marker::PhantomData,
        }
    }
}

// Returns source and target tensors with shapes:
// [batch_size, sequence_length, input_size]
// [batch_size, output_size]
impl<B: Backend> Batcher<B, StockTimeSeriesItemSample, (Tensor<B, 3, Float>, Tensor<B, 2, Float>)>
    for StockTimeSeriesBatcher<B>
{
    /// The `batch` method now receives a `Vec` of our `StockTimeSeriesItemSample`s.
    fn batch(
        &self,
        items: Vec<StockTimeSeriesItemSample>,
        device: &B::Device,
    ) -> (Tensor<B, 3, Float>, Tensor<B, 2, Float>) {
        let batch_size: usize = items.len();
        // Get sequence length from the first item. Assumes all items are the same length.
        let sequence_length: usize = items.first().map_or(0, |item| item.inputs.len());

        if sequence_length == 0 {
            // Handle empty batch
            let inputs_shape: Shape = Shape::new([batch_size, 0, INPUT_SIZE]);
            let targets_shape: Shape = Shape::new([batch_size, TARGET_SIZE]);
            return (
                Tensor::<B, 3, Float>::from_data(
                    TensorData::new(Vec::<f32>::new(), inputs_shape).convert::<f32>(),
                    device,
                ),
                Tensor::<B, 2, Float>::from_data(
                    TensorData::new(Vec::<f32>::new(), targets_shape).convert::<f32>(),
                    device,
                ),
            );
        }

        let mut inputs_data: Vec<f32> =
            Vec::with_capacity(batch_size * sequence_length * INPUT_SIZE);
        let mut targets_data: Vec<f32> = Vec::with_capacity(batch_size * TARGET_SIZE);

        for item in items {
            // Flatten the sequence of inputs
            // item.inputs is Vec<[f32; INPUT_SIZE]> of length `sequence_length`
            for input_features in item.inputs {
                inputs_data.extend_from_slice(&input_features);
            }
            // Add the target features
            targets_data.extend_from_slice(&item.target);
        }

        // Create input tensor: [batch_size, sequence_length, input_size]
        let inputs_shape: Shape = Shape::new([batch_size, sequence_length, INPUT_SIZE]);
        let inputs_tensor_data: TensorData = TensorData::new(inputs_data, inputs_shape.clone());
        // Specify <B, 3, Float> to match the tensor dimensions and type
        let inputs_tensor: Tensor<B, 3, Float> =
            Tensor::<B, 3, Float>::from_data(inputs_tensor_data.convert::<f32>(), device);

        // Create target tensor: [batch_size, target_size]
        let targets_shape: Shape = Shape::new([batch_size, TARGET_SIZE]);
        let targets_tensor_data: TensorData = TensorData::new(targets_data, targets_shape.clone());
        // Specify <B, 2, Float>
        let targets_tensor: Tensor<B, 2, Float> =
            Tensor::<B, 2, Float>::from_data(targets_tensor_data.convert::<f32>(), device);

        (inputs_tensor, targets_tensor)
    }
}
