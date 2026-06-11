# GRU Time-Series Forecasting with Burn

A complete implementation of GRU-based recurrent neural network architectures for
multivariate time-series forecasting using the [Burn](https://burn.dev) deep learning
framework (Rust).

Train a stacked GRU model on historical GOOGL stock-price data with:

```sh
cargo run --example train_stock_gru --release
```

## Overview

This project demonstrates how to design and implement a complete custom RNN architecture
using GRU layers, including:

- **Shape transformation** — how to project GRU outputs to match target dimensions
- **Complete model architectures** — ready-to-use RNN models with all recommended components
- **Training pipeline** — loss computation, batching, and Burn `Learner` integration
- **Best practices** — hyperparameter selection, component choices, and training tips

## Quick Start

### The Core Problem: Shape Transformation

When using a GRU layer, the output shape is `[batch_size, sequence_length, hidden_size]`,
but the target typically has shape `[batch_size, sequence_length, output_size]` where
`output_size ≠ hidden_size`.

**Solution**: add a **linear output projection layer** after the GRU — exactly what
`CustomGruRnn` and `StackedGruRnn` do internally.

```rust
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::tensor::{Float, Tensor};
use burn_gru_time_series_forecasting::gru_model::{StackedGruRnn, StackedGruRnnConfig};

type MyBackend = Wgpu<f32, i32>;

let device = WgpuDevice::default();

let model: StackedGruRnn<MyBackend> = StackedGruRnnConfig::new(
    4,    // input_size  — Open, High, Low, Close
    16,   // hidden_size
    4,    // output_size — predict next-day Open, High, Low, Close
    true, // bias
    0.1,  // dropout
    Initializer::XavierNormal { gain: 1.0 },
)
.init(&device);

// Sequence-to-sequence: [batch=2, seq=10, input=4] → [batch=2, seq=10, output=4]
let x: Tensor<MyBackend, 3, Float> = Tensor::zeros([2, 10, 4], &device);
let seq_out = model.forward(x.clone());   // [2, 10, 4]
let last_out = model.forward_last(x);     // [2, 4]
```

## Architecture Components

### Required Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Linear Output Layer** | Project hidden states to output dimensions | `LinearConfig::new(hidden_size, output_size)` |

### Recommended Components

| Component | Purpose | Typical Value |
|-----------|---------|---------------|
| **Dropout** | Regularization, prevents overfitting — see note | 0.2–0.3 |
| **Multiple Layers** | Increased model capacity | 2 layers |
| **Weight Initialization** | Stable training | Xavier Uniform/Normal |
| **Hidden State Init** | Zero-initialize per sequence | Zero initialization |

> **Dropout note:** Applying a different dropout mask at each time step disrupts the GRU's
> temporal memory and degrades performance. The theoretically correct approach is
> *variational dropout* — the same mask applied at every time step within a sequence
> (Gal & Ghahramani, 2016). Rates of 0.2–0.3 are standard in the financial-forecasting
> literature; rates above 0.4 tend to underfit on small datasets.

### Optional Advanced Components

- **Layer Normalization**: Normalizes across the *feature dimension* at each time step
  (not across layers), stabilizing training and reducing sensitivity to weight
  initialization. Particularly valuable for GRU models, where batch normalization is
  impractical due to variable sequence lengths. Validated for recurrent architectures by
  Ba et al. (2016).

- **Bidirectional Processing**: Reads sequences in both directions to capture context
  from past and future positions. **Important caveat for forecasting:** bidirectional
  GRUs are appropriate only when the full input window is observed offline (e.g.,
  classification over a fixed-length historical window). They are not suitable for
  online or streaming forecasting where future time steps are unavailable at inference
  time, and must never be applied to the prediction horizon itself.

- **Attention Mechanism**: Computes a weighted combination of encoder hidden states,
  allowing the model to focus on the most relevant time steps when producing each output.
  The canonical GRU+attention formulation is the additive scoring mechanism of Bahdanau
  et al. (2015). Qin et al. (2017) apply a dual-stage variant to financial time-series
  forecasting with strong empirical results.

- **Residual Connections**: Address *depth-wise* gradient degradation across stacked GRU
  layers by providing shortcut paths between layers (He et al., 2016). Note that this
  solves a different problem than GRU gating: GRU gates address *temporal* vanishing
  gradients during backpropagation through time (BPTT); residual connections address
  *depth-wise* gradient flow across stacked layers. Both may be present simultaneously
  in a deep multi-layer GRU.

- **Input Projection Layer**: A trainable linear layer applied before the GRU to map raw
  features into a different internal dimension. For continuous-valued time-series inputs
  (OHLCV prices), this is mathematically equivalent to the GRU's own input weight matrix
  and provides no additional representational capacity. It is most beneficial when
  combining heterogeneous continuous and discrete features, or when projecting from a
  high-dimensional input space.

## Usage Examples

### Example 1: Using CustomGruRnn

```rust
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::tensor::{Float, Tensor};
use burn_gru_time_series_forecasting::gru_model::{CustomGruRnn, CustomGruRnnConfig};

type MyBackend = Wgpu<f32, i32>;

let device = WgpuDevice::default();

let model: CustomGruRnn<MyBackend> = CustomGruRnnConfig::new(
    3,    // input_size
    16,   // hidden_size
    3,    // output_size
    true, // bias
    true, // reset_after (Keras-compatible gate ordering)
    0.2,  // dropout
    Initializer::XavierUniform { gain: 1.0 },
)
.init(&device);

let x: Tensor<MyBackend, 3, Float> = Tensor::ones([2, 5, 3], &device);

// Sequence-to-sequence output matches target shape automatically
let out = model.forward(x.clone(), None);   // [2, 5, 3]

// Sequence-to-one: last time step only
let last = model.forward_last(x, None);     // [2, 3]
```

### Example 2: Multi-Layer Stacked GRU

```rust
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::tensor::{Float, Tensor};
use burn_gru_time_series_forecasting::gru_model::{StackedGruRnn, StackedGruRnnConfig};

type MyBackend = Wgpu<f32, i32>;

let device = WgpuDevice::default();

let model: StackedGruRnn<MyBackend> = StackedGruRnnConfig::new(
    3,    // input_size
    32,   // hidden_size
    3,    // output_size
    true, // bias
    0.3,  // dropout (applied after each GRU layer)
    Initializer::XavierNormal { gain: 1.0 },
)
.init(&device);

let x: Tensor<MyBackend, 3, Float> = Tensor::ones([4, 8, 3], &device);

let out  = model.forward(x.clone());   // [4, 8, 3]
let last = model.forward_last(x);      // [4, 3]
```

### Example 3: Dataloading

The input data is a time series of GOOGL stock prices from 2006 to 2017. For data
provenance see [CITATION](data/GOOGL-stock-time-series/CITATION.md).

```rust
use std::path::PathBuf;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn_gru_time_series_forecasting::time_series_dataset::{
    StockTimeSeriesBatcher, StockTimeSeriesItemSample, WindowedStockTimeSeriesDataset,
};

type MyBackend = Wgpu<f32, i32>;

let csv_file = PathBuf::from("data/GOOGL-stock-time-series/GOOGL_2006-01-01_to_2018-01-01.csv");

const SEQUENCE_LENGTH: usize = 8;
const SPLIT: (f32, f32, f32) = (0.7, 0.2, 0.1);

let train_dataset = WindowedStockTimeSeriesDataset::new(
    &csv_file, "train", SPLIT, SEQUENCE_LENGTH,
)
.expect("Failed to create training dataset");

let valid_dataset = WindowedStockTimeSeriesDataset::new(
    &csv_file, "valid", SPLIT, SEQUENCE_LENGTH,
)
.expect("Failed to create validation dataset");

// Each sample: inputs [seq=8, 4 OHLC features], target [4 OHLC features]
let item: StockTimeSeriesItemSample = train_dataset.get(0).unwrap();

let device = WgpuDevice::default();

let dataloader_train = DataLoaderBuilder::new(StockTimeSeriesBatcher::<MyBackend>::new())
    .batch_size(16)
    .num_workers(1)
    .build(train_dataset);

let dataloader_valid = DataLoaderBuilder::new(StockTimeSeriesBatcher::<MyBackend>::new())
    .batch_size(16)
    .num_workers(1)
    .build(valid_dataset);
```

## Running the Examples

### Training Example

Trains on GOOGL historical data and writes weights and config to
`models/stock-time-series-gru/`:

```bash
cargo run --example train_stock_gru --release
```

## Project Structure

```
burn-gru-time-series-forecasting/
├── src/
│   ├── lib.rs                      # Crate root and module exports
│   ├── gru_model.rs                # GRU model architectures
│   │   ├── CustomGruRnn            # Single-layer: GRU + dropout + linear + Tanh
│   │   └── StackedGruRnn           # Two-layer stacked model
│   ├── time_series_dataset.rs      # CSV loading, sliding-window dataset, batcher
│   └── time_series_training.rs     # TrainStep/InferenceStep, Learner wiring, train()
├── examples/
│   └── train_stock_gru.rs          # End-to-end training on GOOGL stock data
├── data/
│   └── GOOGL-stock-time-series/    # Input CSV and citation
├── ARCHITECTURE.md                 # Detailed architecture and design guide
└── README.md                       # This file
```

## Key Features

### CustomGruRnn

Single-layer GRU with:
- ✅ GRU layer for sequence processing
- ✅ Dropout for regularization
- ✅ Linear output projection (handles shape mismatch between hidden and output size)
- ✅ Tanh activation on output
- ✅ `forward()` — full sequence output `[batch, seq, output]`
- ✅ `forward_last()` — last timestep only `[batch, output]`

### StackedGruRnn

Two-layer stacked GRU with:
- ✅ Two GRU layers with per-layer dropout
- ✅ Linear output projection
- ✅ Increased representational capacity for complex temporal patterns
- ✅ `forward()` — full sequence output `[batch, seq, output]`
- ✅ `forward_last()` — last timestep only `[batch, output]`

## Common Use Cases

### Sequence-to-Sequence (Multi-Step Forecasting)

```rust
let predictions = model.forward(x);     // [batch, seq, output]
// All time steps predicted; use a loss over the full sequence
```

### Sequence-to-One (Next-Step Prediction)

```rust
let prediction = model.forward_last(x); // [batch, output]
// Only the final time step output retained; typical for one-step-ahead forecasting
```

## Recommended Hyperparameters

| Parameter | Small Model | Medium Model | Large Model |
|-----------|-------------|--------------|-------------|
| Hidden Size | 64–128 | 128–256 | 256–512 |
| Num Layers | 1–2 | 2–3 | 2–3 ⁽¹⁾ |
| Dropout | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 ⁽²⁾ |
| Learning Rate | 1e-3 | 5e-4 | 1e-4 |

> ⁽¹⁾ **Layer depth:** four or more layers require residual connections between layers
> and layer normalization within each GRU cell to avoid depth-wise gradient degradation
> (Pascanu et al., 2013; He et al., 2016). For standard OHLCV inputs with limited data,
> 2–3 layers is the empirically validated ceiling — additional layers rarely improve
> financial time-series results and increase training instability (Greff et al., 2017;
> Li et al., 2019). Add gradient clipping (max norm 1.0–5.0) when using 3+ layers.

> ⁽²⁾ **Dropout ceiling for financial data:** dropout above 0.4 is common in NLP
> contexts with large corpora and variational (same-mask) dropout. For financial
> time-series — which have a low signal-to-noise ratio and limited sample sizes —
> the practical upper bound is 0.35–0.4; higher rates reliably cause underfitting
> (Rangapuram et al., 2018; Gal & Ghahramani, 2016).

**Recommended for all configurations:**
- Variational dropout (same mask across time steps — Gal & Ghahramani, 2016)
- Xavier weight initialization with `gain = 1.0` for tanh-gated layers
- Zero hidden-state initialization per independent sequence window
- Adam optimizer (Kingma & Ba, 2014) with a learning rate scheduler
  (cosine annealing or ReduceLROnPlateau)

**Optional advanced techniques (see ARCHITECTURE.md):**
- Layer normalization — recommended at 3+ layers
- Bidirectional processing — offline classification only; not for autoregressive forecasting
- Attention mechanisms (Bahdanau et al., 2015; Qin et al., 2017)
- Residual connections between stacked layers (He et al., 2016)

## References

- [Cho et al. (2014) — Learning Phrase Representations using RNN Encoder-Decoder (GRU)](https://arxiv.org/abs/1406.1078)
- [Chung et al. (2014) — Empirical Evaluation of Gated Recurrent Neural Networks](https://arxiv.org/abs/1412.3555)
- [Bahdanau et al. (2015) — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Ba et al. (2016) — Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Gal & Ghahramani (2016) — A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
- [He et al. (2016) — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Pascanu et al. (2013) — On the Difficulty of Training Recurrent Neural Networks](https://proceedings.mlr.press/v28/pascanu13.html)
- [Greff et al. (2017) — LSTM: A Search Space Odyssey](https://doi.org/10.1109/TNNLS.2016.2582924)
- [Qin et al. (2017) — A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)
- [Fischer & Krauss (2018) — Deep Learning with LSTM Networks for Financial Market Predictions](https://doi.org/10.1016/j.ejor.2017.11.054)
- [Rangapuram et al. (2018) — Deep State Space Models for Time Series Forecasting](https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html)
- [Burn Deep Learning Framework](https://burn.dev)
- [Stanford RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

## License

This project follows the workspace license configuration.

## Contributing

Contributions are welcome. Please ensure:
- All examples compile and run
- Code follows Rust best practices
- Documentation is updated accordingly

---

Built with [Burn](https://burn.dev) — a flexible deep learning framework for Rust
