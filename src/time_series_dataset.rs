//! Dataset and batching utilities for stock time-series forecasting with a GRU model.
//!
//! This module provides the data pipeline that sits between raw CSV files and the
//! Burn training loop. It contains three public types:
//!
//! - [`StockTimeSeriesItem`] — a single row deserialised from the CSV file.
//! - [`WindowedStockTimeSeriesDataset`] — a [`burn::data::dataset::Dataset`] that
//!   applies a sliding-window over the raw rows and exposes each window as a
//!   [`StockTimeSeriesItemSample`].
//! - [`StockTimeSeriesBatcher`] — a [`burn::data::dataloader::batcher::Batcher`] that
//!   collates a `Vec<StockTimeSeriesItemSample>` into a pair of rank-3 input tensor
//!   `[batch, seq, features]` and rank-2 target tensor `[batch, targets]`.
//!
//! # When to use this module
//!
//! Use [`WindowedStockTimeSeriesDataset`] when you need a train / test / validation
//! split of a CSV that contains the columns `Date`, `Open`, `High`, `Low`, `Close`,
//! `Volume`, and `Name`. Pair it with [`StockTimeSeriesBatcher`] to feed a Burn
//! [`DataLoader`](burn::data::dataloader::DataLoader).
//!
//! # Example
//!
//! ```no_run
//! use std::path::PathBuf;
//! use burn_gru_time_series_forecasting::time_series_dataset::{
//!     StockTimeSeriesBatcher, WindowedStockTimeSeriesDataset,
//! };
//!
//! let path = PathBuf::from("data/all_stocks_5yr.csv");
//! let sequence_length = 30;
//! let split_sizes = (0.7, 0.15, 0.15);
//!
//! let train = WindowedStockTimeSeriesDataset::new(
//!     &path,
//!     "train",
//!     split_sizes,
//!     sequence_length,
//! )
//! .expect("failed to build training dataset");
//!
//! println!("training windows: {}", burn::data::dataset::Dataset::len(&train));
//! ```

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::Backend;
use burn::tensor::{Float, Shape, Tensor, TensorData};
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// Error type returned by dataset construction and CSV parsing operations.
///
/// Wraps the underlying cause as a human-readable string so callers do not need
/// to depend on `csv` or `std::io` error types directly. Implements
/// [`std::error::Error`] and converts from [`io::Error`] and [`csv::Error`]
/// via [`From`].
#[derive(Debug)]
pub struct DatasetError {
    // Private: callers observe this value through the `Display` impl.
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

impl Error for DatasetError {}

/// Converts an [`io::Error`] into a [`DatasetError`] by capturing its message.
///
/// Allows `?` to be used on file-open operations inside dataset constructors
/// that return `Result<_, DatasetError>`.
impl From<io::Error> for DatasetError {
    fn from(error: io::Error) -> Self {
        DatasetError {
            details: error.to_string(),
        }
    }
}

/// Converts a [`csv::Error`] into a [`DatasetError`] by capturing its message.
///
/// Allows `?` to be used on CSV deserialization calls inside dataset constructors
/// that return `Result<_, DatasetError>`.
impl From<csv::Error> for DatasetError {
    fn from(error: csv::Error) -> Self {
        DatasetError {
            details: error.to_string(),
        }
    }
}

/// Represents one CSV row of daily stock OHLCV data.
///
/// Field names match the CSV column headers through `serde` rename attributes.
/// `volume` and `name` are kept as `String` because the source data may contain
/// non-numeric or ticker-symbol values.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StockTimeSeriesItem {
    /// Trading date, e.g. `"2013-02-08"`.
    #[serde(rename = "Date")]
    pub date: String,
    /// Opening price for the trading day.
    #[serde(rename = "Open")]
    pub open: f32,
    /// Highest price reached during the trading day.
    #[serde(rename = "High")]
    pub high: f32,
    /// Lowest price reached during the trading day.
    #[serde(rename = "Low")]
    pub low: f32,
    /// Closing price for the trading day.
    #[serde(rename = "Close")]
    pub close: f32,
    /// Trading volume as a raw string from the CSV source.
    ///
    /// Kept as `String` because some data providers include formatting characters
    /// (commas, suffixes) that prevent direct `f32` parsing.
    #[serde(rename = "Volume")]
    pub volume: String,
    /// Ticker symbol identifying the stock, e.g. `"AAPL"`.
    #[serde(rename = "Name")]
    pub name: String,
}

/// Number of input features extracted from each [`StockTimeSeriesItem`]:
/// `Open`, `High`, `Low`, `Close`.
const INPUT_SIZE: usize = 4;

/// Number of target values predicted for the next day:
/// `Open`, `High`, `Low`, `Close`.
const TARGET_SIZE: usize = 4;

/// One training sample consisting of an input sequence and a single-step target.
///
/// `inputs` holds `sequence_length` feature arrays, each of length `INPUT_SIZE`.
/// `target` holds the `TARGET_SIZE` values for the day immediately following the
/// input sequence.
#[derive(Clone, Debug)]
pub struct StockTimeSeriesItemSample {
    /// Ordered sequence of per-day OHLC feature arrays used as the model input.
    ///
    /// Length equals the `sequence_length` passed to
    /// [`WindowedStockTimeSeriesDataset::new`]. Each element is
    /// `[open, high, low, close]` for one calendar day.
    pub inputs: Vec<[f32; INPUT_SIZE]>,
    /// OHLC values for the day immediately following the input sequence.
    ///
    /// Serves as the supervised label: `[open, high, low, close]` of the day
    /// the model is asked to forecast.
    pub target: [f32; TARGET_SIZE],
}

/// A sliding-window [`Dataset`] over daily stock price data loaded from a CSV file.
///
/// Reads the CSV once at construction time, selects the requested split, and then
/// exposes every valid window of length `sequence_length` as a
/// [`StockTimeSeriesItemSample`]. The number of available samples is
/// `split_item_count - sequence_length`.
pub struct WindowedStockTimeSeriesDataset {
    /// Raw rows belonging to the selected split.
    items: Vec<StockTimeSeriesItem>,
    /// Number of consecutive rows that form one input sequence.
    sequence_length: usize,
}

impl WindowedStockTimeSeriesDataset {
    /// Reads a CSV file and constructs the dataset for a single named split.
    ///
    /// `file_path` must point to an existing CSV whose headers include `Date`,
    /// `Open`, `High`, `Low`, `Close`, `Volume`, and `Name`.
    ///
    /// `split` selects which portion of the data to use. Accepted values are
    /// `"train"`, `"test"`, `"valid"`, and `"validation"`.
    ///
    /// `split_size` is a tuple `(train, test, valid)` of proportions. The values
    /// need not sum to exactly 1.0; if they do not, each proportion is normalised
    /// by their sum before slicing.
    ///
    /// `sequence_length` controls how many consecutive rows form a single input
    /// window. The split must contain strictly more rows than `sequence_length`.
    ///
    /// # Errors
    ///
    /// Returns an error if `file_path` does not exist, the CSV cannot be parsed,
    /// all proportions sum to zero, `split` is not a recognised name, or the
    /// chosen split contains no more rows than `sequence_length`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::PathBuf;
    /// use burn_gru_time_series_forecasting::time_series_dataset::WindowedStockTimeSeriesDataset;
    ///
    /// let dataset = WindowedStockTimeSeriesDataset::new(
    ///     &PathBuf::from("data/all_stocks_5yr.csv"),
    ///     "train",
    ///     (0.7, 0.15, 0.15),
    ///     30,
    /// )
    /// .unwrap();
    /// ```
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
        }

        let train_count = (total_items as f32 * train_prop).floor() as usize;
        let test_count = (total_items as f32 * test_prop).floor() as usize;

        let train_end = train_count.min(total_items);
        let test_end = (train_end + test_count).min(total_items);
        let valid_end = total_items;

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

    // Selects only the four OHLC price fields the model consumes as inputs,
    // discarding date, volume, and ticker columns.
    fn item_to_features(item: &StockTimeSeriesItem) -> [f32; INPUT_SIZE] {
        [item.open, item.high, item.low, item.close]
    }

    // Selects the four OHLC price fields used as the supervised target.
    // Kept as a separate function from `item_to_features` so the input and
    // target feature sets can diverge independently in the future.
    fn item_to_target(item: &StockTimeSeriesItem) -> [f32; TARGET_SIZE] {
        [item.open, item.high, item.low, item.close]
    }
}

impl Dataset<StockTimeSeriesItemSample> for WindowedStockTimeSeriesDataset {
    /// Returns the sample whose input window starts at `index`, or `None` if
    /// `index` is out of range.
    ///
    /// The window spans rows `index..index + sequence_length` as inputs and
    /// row `index + sequence_length` as the target. Returns `None` when the
    /// window would extend past the end of the split, consistent with the count
    /// returned by [`Self::len`].
    fn get(&self, index: usize) -> Option<StockTimeSeriesItemSample> {
        if index + self.sequence_length >= self.items.len() {
            return None;
        }

        let inputs: Vec<[f32; INPUT_SIZE]> = (index..index + self.sequence_length)
            .map(|i| Self::item_to_features(&self.items[i]))
            .collect();

        let target = Self::item_to_target(&self.items[index + self.sequence_length]);

        Some(StockTimeSeriesItemSample { inputs, target })
    }

    /// Returns the number of complete windows available in this split.
    ///
    /// Equal to `split_item_count - sequence_length`. The last `sequence_length`
    /// rows cannot serve as sequence starts because they have no subsequent target.
    fn len(&self) -> usize {
        self.items.len().saturating_sub(self.sequence_length)
    }
}

/// Collates [`StockTimeSeriesItemSample`] values into batched tensors for training.
///
/// Produces an input tensor of shape `[batch_size, sequence_length, INPUT_SIZE]`
/// and a target tensor of shape `[batch_size, TARGET_SIZE]`. Both tensors use
/// `f32` element type and are placed on the device supplied to [`Batcher::batch`].
///
/// `B` is the Burn backend (e.g. `Wgpu`, `NdArray`, `Autodiff<Wgpu>`). The struct
/// holds no runtime state; all backend specialisation is carried through the
/// `PhantomData` marker.
pub struct StockTimeSeriesBatcher<B: Backend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for StockTimeSeriesBatcher<B> {
    /// Creates a [`StockTimeSeriesBatcher`] using default construction.
    ///
    /// Delegates to [`StockTimeSeriesBatcher::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> StockTimeSeriesBatcher<B> {
    /// Creates a new batcher for backend `B`.
    ///
    /// The batcher holds no configuration; all shape information is inferred
    /// from the samples passed to [`Batcher::batch`] at runtime.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn::backend::NdArray;
    /// use burn_gru_time_series_forecasting::time_series_dataset::StockTimeSeriesBatcher;
    ///
    /// let batcher = StockTimeSeriesBatcher::<NdArray>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, StockTimeSeriesItemSample, (Tensor<B, 3, Float>, Tensor<B, 2, Float>)>
    for StockTimeSeriesBatcher<B>
{
    /// Collates a vector of samples into an `(inputs, targets)` tensor pair.
    ///
    /// The returned input tensor has shape `[batch_size, sequence_length, INPUT_SIZE]`
    /// and the returned target tensor has shape `[batch_size, TARGET_SIZE]`.
    /// Both are placed on `device`. If `items` is empty the tensors have a
    /// zero sequence-length dimension and contain no data. `sequence_length` is
    /// inferred from the first sample; all samples in a batch must share the same
    /// sequence length, which is guaranteed when they originate from the same
    /// [`WindowedStockTimeSeriesDataset`].
    fn batch(
        &self,
        items: Vec<StockTimeSeriesItemSample>,
        device: &B::Device,
    ) -> (Tensor<B, 3, Float>, Tensor<B, 2, Float>) {
        let batch_size: usize = items.len();
        let sequence_length: usize = items.first().map_or(0, |item| item.inputs.len());

        if sequence_length == 0 {
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
            for input_features in item.inputs {
                inputs_data.extend_from_slice(&input_features);
            }
            targets_data.extend_from_slice(&item.target);
        }

        let inputs_shape: Shape = Shape::new([batch_size, sequence_length, INPUT_SIZE]);
        let inputs_tensor: Tensor<B, 3, Float> = Tensor::<B, 3, Float>::from_data(
            TensorData::new(inputs_data, inputs_shape).convert::<f32>(),
            device,
        );

        let targets_shape: Shape = Shape::new([batch_size, TARGET_SIZE]);
        let targets_tensor: Tensor<B, 2, Float> = Tensor::<B, 2, Float>::from_data(
            TensorData::new(targets_data, targets_shape).convert::<f32>(),
            device,
        );

        (inputs_tensor, targets_tensor)
    }
}
