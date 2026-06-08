# Custom GRU RNN Architecture with Burn

A comprehensive guide and implementation for building custom Recurrent Neural Network (RNN) architectures using Gated Recurrent Units (GRU) in the [Burn](https://burn.dev) deep learning framework (Rust).

The full training and inference example can be run with:
```sh
cargo run --example burn_gru
```

## Overview

This project demonstrates how to design and implement a complete custom RNN architecture using GRU layers, including:

- **Shape transformation**: How to project GRU outputs to match target dimensions
- **Complete model architectures**: Ready-to-use RNN models with all recommended components
- **Training examples**: Loss computation, forward passes, and evaluation metrics
- **Best practices**: Hyperparameter selection, component choices, and training tips

## Quick Start

### The Core Problem: Shape Transformation

When using a GRU layer, the output shape is `[batch_size, sequence_length, hidden_size]`, but your target often has shape `[batch_size, sequence_length, output_size]` where `output_size ≠ hidden_size`.

**Solution**: Add a **Linear output projection layer**

```rust
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Float, Tensor};

type MyBackend = Wgpu<f32, i32>;
let device = WgpuDevice::default();

// ========== Complete Custom RNN Architecture Example ==========
println!("\n=== Complete Custom RNN Architecture ===");

// Architecture components:
// 1. Input layer (optional normalization/embedding)
// 2. GRU layers (can stack multiple)
// 3. Output projection layer
// 4. Optional: Dropout, LayerNorm, Attention

let input_size = 3;  // number of features in each item in an input sequence
let hidden_size = 16;
let output_size = 3;  // number of features in each item in an output sequence

// Component 1: GRU layer
let gru_model: Gru<MyBackend> = GruConfig::new(input_size, hidden_size, true)
    .with_reset_after(true)
    .with_initializer(Initializer::XavierUniform { gain: 1.0 })
    .init(&device);

// Component 2: Output projection layer (REQUIRED)
let output_projection: Linear<MyBackend> = LinearConfig::new(hidden_size, output_size)
    .with_bias(true)
    .init(&device);

// Forward pass
let x_input: Tensor<MyBackend, 3, Float> = Tensor::ones([2, 10, input_size], &device); // [batch=2, seq_len=10, input_size=3]
let h_init: Tensor<MyBackend, 2, Float> = Tensor::zeros([2, hidden_size], &device); // [batch=2, hidden_size=16]

// GRU forward
let gru_out = gru_model.forward(x_input, Some(h_init)); // [2, 10, 16]
println!("GRU output: {:?}", gru_out.shape());

// Project to output size
let final_output = output_projection.forward(gru_out);
println!("Final output: {:?}", final_output.shape());
}
```

## Architecture Components

### Required Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Linear Output Layer** | Project hidden states to output dimensions | `LinearConfig::new(hidden_size, output_size)` |

### Recommended Components

| Component | Purpose | Typical Value |
|-----------|---------|---------------|
| **Dropout** | Regularization, prevents overfitting | 0.2 - 0.3 |
| **Multiple Layers** | Increased model capacity | 2 layers |
| **Weight Initialization** | Stable training | Xavier Uniform/Normal |
| **Hidden State Init** | Control initial processing | Zero initialization |

### Optional Advanced Components

- **Layer Normalization**: Stabilizes training by normalizing activations across layers, improving convergence speed, and reducing sensitivity to initialization.
- **Bidirectional Processing**: Enables capturing context from both past and future inputs, enhancing overall model comprehension and performance on sequential data tasks.
- **Attention Mechanism**: Allows the model to focus on relevant parts of input data when generating outputs, improving performance and enabling better contextual understanding.
- **Residual Connections**: Help alleviate vanishing gradients by providing shortcut paths for gradient flow during backpropagation, allowing deeper networks to train effectively.
- **Embedding Layer**: Transform input data into dense vectors, allowing the model to capture semantic relationships between inputs and improve performance.

## Usage Examples

### Example 1: Using CustomGruRnn

```rust
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::nn::Initializer;
use burn::tensor::{Float, Tensor};
use burn_gru::gru_model::{CustomGruRnn, CustomGruRnnConfig};

type MyBackend = Wgpu<f32, i32>;
let device = WgpuDevice::default();

// ========== Using CustomGruRnn Model ==========
println!("\n=== CustomGruRnn Model Example ===");

let custom_model: CustomGruRnn<MyBackend> = CustomGruRnnConfig::new(
    3,    // input_size
    16,   // hidden_size
    3,    // output_size (matches target)
    true, // bias
    true, // reset_after
    0.2,  // dropout
    Initializer::XavierUniform { gain: 1.0 },
)
.init(&device);

let x_custom: Tensor<MyBackend, 3, Float> = Tensor::ones([2, 5, 3], &device); // [batch=2, seq=5, input=3]
let y_target: Tensor<MyBackend, 3, Float> = Tensor::zeros([2, 5, 3], &device); // [batch=2, seq=5, output=3]

// Forward pass - output shape automatically matches target
let prediction = custom_model.forward(x_custom.clone(), None);
println!("CustomGruRnn input shape: {:?}", x_custom.shape());
println!("CustomGruRnn output shape: {:?}", prediction.shape());
println!("Target shape: {:?}", y_target.shape());
assert_eq!(prediction.shape(), y_target.shape(), "Shapes must match!");
println!("✓ Output shape matches target shape!");

// Using forward_last for sequence classification
let last_output = custom_model.forward_last(x_custom.clone(), None);
println!("Last time step output shape: {:?}", last_output.shape()); // [batch=2, output=3]
```

### Example 2: Multi-Layer Stacked GRU

```rust
use burn_gru::gru_model::{StackedGruRnn, StackedGruRnnConfig};

// ========== Using StackedGruRnn Model ==========
println!("\n=== StackedGruRnn Model Example (Multi-layer) ===");

let stacked_model: StackedGruRnn<MyBackend> = StackedGruRnnConfig::new(
    3,    // input_size (number of features in the input sequence length)
    32,   // hidden_size
    3,    // output_size (number of features in the output sequence length)
    true, // bias
    0.3,  // dropout
    Initializer::XavierNormal { gain: 1.0 },
)
.init(&device);

let x_stacked: Tensor<MyBackend, 3, Float> = Tensor::ones([4, 8, 3], &device); // [batch_size, seq_len, input_size]
let stacked_output =
    stacked_model.forward(x_stacked.clone());
println!("StackedGruRnn input shape: {:?}", x_stacked.shape());
println!("StackedGruRnn output shape: {:?}", stacked_output.shape());
println!("✓ Multi-layer GRU provides more representational power!");

// Using forward_last for sequence classification
let last_output = stacked_model.forward_last(x_stacked.clone());
println!("Last time step output shape: {:?}", last_output.shape()); // [batch=2, output=3]
```

### Example 3: Dataloading
The input data for this example is a time series of stock prices for "GOOGL" from 2006 to 2017. For more information about the data, please see [CITATION](data/GOOGL-stock-time-series/CITATION.md).

```rust
// ========== Data Loading ==========
    println!("\n=== Data Loading Example ===");
    let csv_file = Path::new("data")
        .join("GOOGL-stock-time-series")
        .join("GOOGL_2006-01-01_to_2018-01-01.csv");

    const SEQUENCE_LENGTH: usize = 8;
    const TRAIN_TEST_VALID_SPLIT: (f32, f32, f32) = (0.7, 0.2, 0.1);
    let train_dataset: WindowedStockTimeSeriesDataset = WindowedStockTimeSeriesDataset::new(
        &csv_file,
        "train",
        TRAIN_TEST_VALID_SPLIT,
        SEQUENCE_LENGTH,
    )
    .expect("Failed to create training dataset");

    let test_dataset: WindowedStockTimeSeriesDataset = WindowedStockTimeSeriesDataset::new(
        &csv_file,
        "test",
        TRAIN_TEST_VALID_SPLIT,
        SEQUENCE_LENGTH,
    )
    .expect("Failed to create testing dataset");

    let valid_dataset: WindowedStockTimeSeriesDataset = WindowedStockTimeSeriesDataset::new(
        &csv_file,
        "valid",
        TRAIN_TEST_VALID_SPLIT,
        SEQUENCE_LENGTH,
    )
    .expect("Failed to create validation dataset");

    // Example: get a single item from the dataset
    let item: StockTimeSeriesItemSample = train_dataset
        .get(0)
        .expect("expected to get a StockTimeSeriesItem");
    // Iterate over the training dataset
    for item in train_dataset.iter() {
        println!("{:?}", item);
    }

    // Create Datalaoders for the training and validation datasets.
    // These are used for training in a custom loop or with burn's Learner.
    let batcher_train = StockTimeSeriesBatcher::<MyBackend>::new();
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1)
        .num_workers(1)
        .build(train_dataset);
    let batcher_valid = StockTimeSeriesBatcher::<MyBackend>::new();
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1)
        .num_workers(1)
        .build(valid_dataset);
```

## Running the Examples

### Training Example
Shows loss computation, evaluation metrics, and training patterns:
```bash
cargo run --example burn_gru
```

## Project Structure

```
burn-gru/
├── src/
│   ├── lib.rs                   # Module exports
│   ├── gru_model.rs             # Custom RNN architectures
│   │   ├── CustomGruRnn         # Single-layer with dropout
│   │   └── StackedGruRnn        # Multi-layer stacked model
│   └── time_series_dataset.rs   # Dataset utilities
│   └── time_series_training.rs  # Training and loss computation
├── examples/
│   ├── burn_gru.rs              # Shape transformation examples
├── ARCHITECTURE.md             # Comprehensive architecture guide
└── README.md                   # This file
```

## Key Features

### CustomGruRnn Model

A complete custom RNN with:
- ✅ GRU layer for sequence processing
- ✅ Dropout for regularization
- ✅ Linear output projection (automatically handles shape matching)
- ✅ `forward()`: Full sequence output `[batch, seq, output]`
- ✅ `forward_last()`: Last timestep only `[batch, output]` (for classification)

### StackedGruRnn Model

A multi-layer architecture with:
- ✅ Multiple stacked GRU layers
- ✅ Dropout between layers
- ✅ Increased representational capacity
- ✅ Better for complex tasks

## Common Use Cases

### Sequence-to-Sequence (Time Series Forecasting)
```rust
let predictions = model.forward(x, None); // [batch, seq, output]
// All timesteps are predicted
```

### Sequence-to-One (Classification)
```rust
let logits = model.forward_last(x, None); // [batch, output]
let predictions = logits.argmax(1);
```

### Many-to-Many (Translation, etc.)
```rust
// Encoder
let encoded = encoder.forward(source_sequence, None);
// Decoder
let decoded = decoder.forward(target_sequence, encoded_state);
```

## Recommended Hyperparameters

| Parameter | Small Model | Medium Model | Large Model |
|-----------|-------------|--------------|-------------|
| Hidden Size | 64-128 | 128-256 | 256-512 |
| Num Layers | 1-2 | 2-3 | 3-4 |
| Dropout | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 |
| Learning Rate | 1e-3 | 5e-4 | 1e-4 |

**Recommended**:
- Dropout layer (0.2-0.3)
- Multiple stacked GRU layers
- Proper weight initialization (Xavier)
- Hidden state management

**Optional Advanced**:
- Layer normalization
- Bidirectional processing
- Attention mechanisms
- Residual connections

## References

- [Burn Deep Learning Framework](https://burn.dev)
- [GRU Paper (Cho et al., 2014)](https://arxiv.org/abs/1406.1078)
- [Stanford RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
- [PyTorch GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

## License

This project follows the workspace license configuration.

## Contributing

Contributions are welcome! Please ensure:
- All examples compile and run
- Code follows Rust best practices
- Documentation is updated accordingly

---

**Built with ❤️ using [Burn](https://burn.dev) - A flexible deep learning framework for Rust**
