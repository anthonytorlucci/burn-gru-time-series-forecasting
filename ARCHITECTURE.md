# Custom GRU RNN Architecture Guide

This document explains how to build a complete custom Recurrent Neural Network (RNN) architecture using Gated Recurrent Units (GRU) in the Burn library (Rust).

## Table of Contents

1. [Problem: Shape Transformation (Line 27)](#problem-shape-transformation)
2. [Required Components](#required-components)
3. [Recommended Components](#recommended-components)
4. [Optional Advanced Components](#optional-advanced-components)
5. [Architecture Examples](#architecture-examples)
6. [Best Practices](#best-practices)

---

## Required Components

### 1. Linear Output Projection Layer ⚠️ REQUIRED

**Purpose:** Transform GRU hidden state to match target output dimensions.

**Implementation:**
```rust
use burn::nn::{Linear, LinearConfig};

let output_layer: Linear<Backend> = LinearConfig::new(hidden_size, output_size)
    .with_bias(true)
    .init(&device);
```

**Why Required:**
- GRU outputs hidden states, not task-specific predictions
- Target dimensions rarely match hidden dimensions
- Provides learnable transformation for task adaptation

---

## Recommended Components

### 2. Dropout Layer 🎯 HIGHLY RECOMMENDED

**Purpose:** Regularization to prevent overfitting.

**Implementation:**
```rust
use burn::nn::{Dropout, DropoutConfig};

let dropout: Dropout = DropoutConfig::new(0.2)  // 20% dropout rate
    .init();
```

**Best Practices:**
- Use dropout rate between 0.1 and 0.5
- Apply after GRU layer(s)
- Higher dropout for larger models
- Disable during inference (automatic in Burn)

### 3. Multiple Stacked GRU Layers 🎯 RECOMMENDED

**Purpose:** Increase model capacity and representational power.

**Implementation:**
```rust
// Layer 1: input_size -> hidden_size
let gru1 = GruConfig::new(input_size, hidden_size, true).init(&device);
let dropout1 = DropoutConfig::new(0.2).init();

// Layer 2: hidden_size -> hidden_size
let gru2 = GruConfig::new(hidden_size, hidden_size, true).init(&device);
let dropout2 = DropoutConfig::new(0.2).init();

// Forward pass
let x = gru1.forward(x, Some(h1));
let x = dropout1.forward(x);
let x = gru2.forward(x, Some(h2));
let x = dropout2.forward(x);
```

**Best Practices:**
- Start with 1-2 layers
- Add dropout between layers
- Each layer processes increasingly abstract features

### 4. Proper Weight Initialization 🎯 RECOMMENDED

**Purpose:** Stable training and faster convergence.

**Implementation:**
```rust
use burn::nn::Initializer;

let gru = GruConfig::new(input_size, hidden_size, bias)
    .with_initializer(Initializer::XavierUniform { gain: 1.0 })
    .init(&device);
```

**Options:**
- `XavierUniform` / `XavierNormal`: Good for most cases
- `KaimingUniform` / `KaimingNormal`: Good with ReLU activations
- `Normal { mean, std }`: Basic Gaussian initialization

### 5. Hidden State Management 🎯 RECOMMENDED

**Purpose:** Control initial state of recurrent processing.

**Implementation:**
```rust
// Option 1: Zero initialization (most common)
let h = Tensor::zeros(Shape::new([batch_size, hidden_size]), &device);

// Option 2: Random initialization
let h = Tensor::random(Shape::new([batch_size, hidden_size]), 
                       Distribution::Normal(0.0, 0.1), &device);

// Option 3: Learned initialization (advanced)
// Create as a parameter in your model struct
```

---

## Optional Advanced Components

### 6. Layer Normalization

**Purpose:** Stabilize training by normalizing activations.

**When to use:** Deep networks (3+ layers) or training instability.

```rust
use burn::nn::{LayerNorm, LayerNormConfig};

let layer_norm = LayerNormConfig::new(hidden_size).init(&device);
let x = layer_norm.forward(x);
```

### 7. Bidirectional Processing

**Purpose:** Process sequence in both forward and backward directions.

**Implementation concept:**
```rust
// Forward GRU
let forward_out = gru_forward.forward(x.clone(), Some(h_forward));

// Backward GRU (reverse sequence)
let x_reversed = reverse_sequence(x);
let backward_out = gru_backward.forward(x_reversed, Some(h_backward));
let backward_out = reverse_sequence(backward_out);

// Concatenate outputs
let bidirectional_out = Tensor::cat(vec![forward_out, backward_out], 2);
```

### 8. Attention Mechanism

**Purpose:** Allow model to focus on relevant parts of input sequence.

**Use case:** Long sequences, sequence-to-sequence tasks.

### 9. Residual Connections

**Purpose:** Enable gradient flow in very deep networks.

**Implementation:**
```rust
let residual = x.clone();
let x = gru.forward(x, Some(h));
let x = x + residual;  // Skip connection
```

### 10. Activation Functions

**Purpose:** Non-linearity after output projection.

**Implementation:**
```rust
use burn::nn::{Relu, Tanh, Sigmoid};

let activation = Tanh::new();
let output = activation.forward(output_projection);
```

---

## Best Practices

### Model Design

1. **Start Simple**: Begin with single-layer GRU + output projection
2. **Add Complexity Gradually**: Add dropout, then more layers, then advanced features
3. **Match Architecture to Task**:
   - **Sequence-to-sequence**: Full output `[batch, seq, output]`
   - **Sequence-to-one**: Extract last timestep `[batch, output]`
   - **Classification**: Add softmax after output layer

### Hyperparameter Selection

| Component | Typical Range | Starting Point |
|-----------|---------------|----------------|
| Hidden Size | 64-512 | 128 |
| Num Layers | 1-4 | 2 |
| Dropout | 0.1-0.5 | 0.2 |
| Learning Rate | 1e-4 to 1e-2 | 1e-3 |

### Training Tips

1. **Gradient Clipping**: Prevent exploding gradients (clip at 1.0-5.0)
2. **Sequence Length**: Start with shorter sequences, increase gradually
3. **Batch Size**: Larger for stable training (16-128)
4. **Validation**: Monitor validation loss to detect overfitting

### Common Patterns

#### Pattern 1: Sequence Classification
```rust
// Use forward_last to get [batch, output_size]
let logits = model.forward_last(x, None);
let predictions = logits.argmax(1);
```

#### Pattern 2: Time Series Forecasting
```rust
// Use full sequence output [batch, seq, output_size]
let predictions = model.forward(x, None);
// predictions[:, -1, :] contains the forecast
```

#### Pattern 3: Sequence-to-Sequence
```rust
// Encoder-decoder architecture
let encoded = encoder.forward(source_sequence, None);
let decoded = decoder.forward(target_sequence, encoded_state);
```

---

## Summary Checklist

When building a custom GRU RNN:

- [x] **REQUIRED**: Linear output projection layer
- [x] **RECOMMENDED**: Dropout layer (0.2-0.3)
- [x] **RECOMMENDED**: Proper weight initialization (Xavier)
- [x] **RECOMMENDED**: Hidden state initialization
- [ ] **OPTIONAL**: Multiple stacked layers (for complex tasks)
- [ ] **OPTIONAL**: Layer normalization (for deep networks)
- [ ] **OPTIONAL**: Bidirectional processing (for context-rich tasks)
- [ ] **OPTIONAL**: Attention mechanism (for long sequences)

---

## References

- [Burn Documentation](https://burn.dev)
- [GRU Paper](https://arxiv.org/abs/1406.1078): "Learning Phrase Representations using RNN Encoder-Decoder"
- [PyTorch GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html): For comparison with other frameworks
