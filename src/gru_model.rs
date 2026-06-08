use burn::config::Config;
use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig, Tanh};
use burn::prelude::Shape;
use burn::tensor::backend::Backend;
use burn::tensor::{Float, Tensor};

// --- Complete Custom RNN Architecture Model
/// A complete custom RNN model with all recommended components:
/// - GRU layer for sequence processing
/// - Dropout for regularization
/// - Linear output projection (REQUIRED to match target shape)
/// - Optional layer normalization
#[derive(Module, Debug)]
pub struct CustomGruRnn<B: Backend> {
    /// Main GRU layer
    gru: Gru<B>,
    /// Dropout for regularization
    dropout: Dropout,
    /// Output projection layer (projects hidden_size -> output_size)
    output_projection: Linear<B>,
    /// Activation function
    activation: Tanh,
}

/// Configuration for the complete custom RNN model
#[derive(Config, Debug)]
pub struct CustomGruRnnConfig {
    /// Input feature size
    input_size: usize,
    /// Hidden state size
    hidden_size: usize,
    /// Output size (must match target dimension)
    output_size: usize,
    /// Whether to use bias in GRU
    bias: bool,
    /// Whether to apply reset gate after matrix multiplication
    reset_after: bool,
    /// Dropout probability (0.0 = no dropout)
    dropout: f64,
    /// Weight initializer
    initializer: Initializer,
}

impl CustomGruRnnConfig {
    /// Initialize the custom GRU RNN model
    pub fn init<B: Backend>(&self, device: &B::Device) -> CustomGruRnn<B> {
        CustomGruRnn {
            gru: GruConfig::new(self.input_size, self.hidden_size, self.bias)
                .with_reset_after(self.reset_after)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            output_projection: LinearConfig::new(self.hidden_size, self.output_size)
                .with_bias(true)
                .init(device),
            activation: Tanh::new(),
        }
    }
}

impl<B: Backend> CustomGruRnn<B> {
    /// Forward pass through the complete RNN architecture
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, sequence_length, input_size]
    /// * `h` - Optional initial hidden state of shape [batch_size, hidden_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, sequence_length, output_size]
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        h: Option<Tensor<B, 2, Float>>,
    ) -> Tensor<B, 3, Float> {
        let batch_size = x.dims()[0];
        let _seq_len = x.dims()[1];

        // Initialize hidden state if not provided
        let h = h.unwrap_or_else(|| {
            Tensor::zeros(Shape::new([batch_size, self.gru.d_hidden]), &x.device())
        });

        // Step 1: GRU layer - processes sequence
        let gru_out = self.gru.forward(x, Some(h)); // [batch_size, seq_len, hidden_size]

        // Step 2: Apply dropout for regularization
        let gru_out = self.dropout.forward(gru_out);

        // Step 3: Reshape for linear layer
        // [batch_size, seq_len, hidden_size] -> [batch_size * seq_len, hidden_size]
        // let hidden_size = gru_out.dims()[2];
        //--let gru_out_flat = gru_out.reshape([batch_size * seq_len, hidden_size]);

        // Step 4: Apply output projection layer (REQUIRED)
        let gru_out = self.output_projection.forward(gru_out); // [batch_size, seq_len, output_size]

        // Step 5: Apply activation function
        self.activation.forward(gru_out)
    }

    /// Forward pass that returns only the last time step output
    /// Useful for sequence-to-one tasks (e.g., sequence classification)
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, sequence_length, input_size]
    /// * `h` - Optional initial hidden state of shape [batch_size, hidden_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, output_size]
    pub fn forward_last(
        &self,
        x: Tensor<B, 3, Float>,
        h: Option<Tensor<B, 2, Float>>,
    ) -> Tensor<B, 2, Float> {
        let output = self.forward(x, h); // [batch_size, seq_len, output_size]
        let batch_size = output.dims()[0];
        let seq_len = output.dims()[1];
        let output_size = output.dims()[2];

        // Extract last time step: select [:, -1, :]
        // Use reshape instead of squeeze to ensure consistent 2D output
        // even when batch_size == 1
        output
            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..output_size])
            .reshape([batch_size, output_size])
    }
}

// --- Multi-Layer Stacked GRU Model (RECOMMENDED for deeper models)
/// A multi-layer stacked GRU model with output projection
#[derive(Module, Debug)]
pub struct StackedGruRnn<B: Backend> {
    /// First GRU layer
    gru_layer1: Gru<B>,
    /// Dropout after first layer
    dropout1: Dropout,
    /// Second GRU layer
    gru_layer2: Gru<B>,
    /// Dropout after second layer
    dropout2: Dropout,
    /// Output projection layer
    output_projection: Linear<B>,
}

/// Configuration for stacked GRU model
#[derive(Config, Debug)]
pub struct StackedGruRnnConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    bias: bool,
    dropout: f64,
    initializer: Initializer,
}

impl StackedGruRnnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StackedGruRnn<B> {
        // For simplicity, implementing 2-layer version
        // In production, you'd use a Vec<Gru<B>> for arbitrary num_layers
        StackedGruRnn {
            gru_layer1: GruConfig::new(self.input_size, self.hidden_size, self.bias)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout1: DropoutConfig::new(self.dropout).init(),
            gru_layer2: GruConfig::new(self.hidden_size, self.hidden_size, self.bias)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout2: DropoutConfig::new(self.dropout).init(),
            output_projection: LinearConfig::new(self.hidden_size, self.output_size)
                .with_bias(true)
                .init(device),
        }
    }
}

impl<B: Backend> StackedGruRnn<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        let batch_size = x.dims()[0];
        let _seq_len = x.dims()[1];

        // Layer 1
        let h1 = Tensor::zeros(
            Shape::new([batch_size, self.gru_layer1.d_hidden]),
            &x.device(),
        );
        let x = self.gru_layer1.forward(x, Some(h1));
        let x = self.dropout1.forward(x);

        // Layer 2
        let h2 = Tensor::zeros(
            Shape::new([batch_size, self.gru_layer2.d_hidden]),
            &x.device(),
        );
        let x = self.gru_layer2.forward(x, Some(h2));
        let x = self.dropout2.forward(x);

        // Output projection
        self.output_projection.forward(x)
    }

    /// Forward pass that returns only the last time step output
    /// Useful for sequence-to-one tasks (e.g., sequence classification)
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, sequence_length, input_size]
    /// * `h` - Optional initial hidden state of shape [batch_size, hidden_size]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, output_size]
    pub fn forward_last(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 2, Float> {
        let output = self.forward(x); // [batch_size, seq_len, output_size]
        let batch_size = output.dims()[0];
        let seq_len = output.dims()[1];
        let output_size = output.dims()[2];

        // Extract last time step: select [:, -1, :]
        // Use reshape instead of squeeze to ensure consistent 2D output
        // even when batch_size == 1
        output
            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..output_size])
            .reshape([batch_size, output_size])
    }
}
