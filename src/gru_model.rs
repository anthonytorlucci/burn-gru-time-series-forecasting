//! GRU-based recurrent model definitions for time-series forecasting.
//!
//! Provides two model variants built on Burn's [`Gru`] primitive:
//!
//! - [`CustomGruRnn`] — single GRU layer with dropout, linear projection, and Tanh activation.
//! - [`StackedGruRnn`] — two stacked GRU layers with per-layer dropout and a linear projection.
//!
//! Both expose a `forward` method (sequence-to-sequence) and a `forward_last` method
//! (sequence-to-one), making them suitable for multi-step prediction and next-step
//! forecasting respectively.
//!
//! # Examples
//!
//! ```no_run
//! use burn::backend::NdArray;
//! use burn::nn::Initializer;
//! use burn_gru_time_series_forecasting::gru_model::{StackedGruRnn, StackedGruRnnConfig};
//!
//! let device = Default::default();
//! let model: StackedGruRnn<NdArray> = StackedGruRnnConfig::new(
//!     4,    // input_size
//!     16,   // hidden_size
//!     4,    // output_size
//!     true, // bias
//!     0.1,  // dropout
//!     Initializer::XavierNormal { gain: 1.0 },
//! )
//! .init(&device);
//! ```

use burn::config::Config;
use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig, Tanh};
use burn::prelude::Shape;
use burn::tensor::backend::Backend;
use burn::tensor::{Float, Tensor};

/// Single-layer GRU model with dropout, linear projection, and Tanh activation.
///
/// Suitable for sequence-to-sequence or sequence-to-one regression tasks. The
/// hidden state is zero-initialised on each forward pass unless an explicit
/// initial state is supplied.
///
/// # Examples
///
/// ```no_run
/// use burn::backend::NdArray;
/// use burn::nn::Initializer;
/// use burn_gru_time_series_forecasting::gru_model::{CustomGruRnn, CustomGruRnnConfig};
///
/// let device = Default::default();
/// let model: CustomGruRnn<NdArray> = CustomGruRnnConfig::new(
///     4, 16, 4, true, false, 0.1,
///     Initializer::XavierNormal { gain: 1.0 },
/// )
/// .init(&device);
/// ```
#[derive(Module, Debug)]
pub struct CustomGruRnn<B: Backend> {
    /// GRU layer for sequence processing.
    gru: Gru<B>,
    /// Dropout applied after the GRU output.
    dropout: Dropout,
    /// Projects hidden states from `hidden_size` to `output_size`.
    output_projection: Linear<B>,
    /// Tanh activation applied after the output projection.
    activation: Tanh,
}

/// Configuration for [`CustomGruRnn`]
#[derive(Config, Debug)]
pub struct CustomGruRnnConfig {
    /// Number of input features per time step.
    input_size: usize,
    /// Dimensionality of the GRU hidden state.
    hidden_size: usize,
    /// Number of output features per time step.
    output_size: usize,
    /// Whether to include bias terms in the GRU gates.
    bias: bool,
    /// When `true`, applies the reset gate after the matrix multiplication (Keras convention).
    reset_after: bool,
    /// Dropout probability applied to GRU outputs; `0.0` disables dropout.
    dropout: f64,
    /// Weight initializer for GRU and linear layers.
    initializer: Initializer,
}

impl CustomGruRnnConfig {
    /// Initialise a [`CustomGruRnn`] model from this configuration
    ///
    /// Both the GRU layer and the output projection are initialised on `device`
    /// using the `initializer` specified in the configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn::backend::NdArray;
    /// use burn::nn::Initializer;
    /// use burn_gru_time_series_forecasting::gru_model::{CustomGruRnn, CustomGruRnnConfig};
    ///
    /// let device = Default::default();
    /// let model: CustomGruRnn<NdArray> = CustomGruRnnConfig::new(
    ///     4, 16, 4, true, false, 0.1,
    ///     Initializer::XavierNormal { gain: 1.0 },
    /// )
    /// .init(&device);
    /// ```
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
    /// Runs a full sequence through the model and returns all time-step outputs.
    ///
    /// Accepts an optional initial hidden state `h`. When `None`, the hidden
    /// state is zero-initialised to shape `[batch_size, hidden_size]`.
    ///
    /// Returns a tensor of shape `[batch_size, sequence_length, output_size]`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use burn::backend::NdArray;
    /// # use burn::nn::Initializer;
    /// # use burn::tensor::Tensor;
    /// # use burn_gru_time_series_forecasting::gru_model::CustomGruRnnConfig;
    /// # let device = Default::default();
    /// # let model = CustomGruRnnConfig::new(4, 16, 4, true, false, 0.0,
    /// #     Initializer::XavierNormal { gain: 1.0 }).init::<NdArray>(&device);
    /// let x: Tensor<NdArray, 3> = Tensor::zeros([2, 10, 4], &device);
    /// let out = model.forward(x, None); // [2, 10, 4]
    /// ```
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        h: Option<Tensor<B, 2, Float>>,
    ) -> Tensor<B, 3, Float> {
        let batch_size = x.dims()[0];

        let h = h.unwrap_or_else(|| {
            Tensor::zeros(Shape::new([batch_size, self.gru.d_hidden]), &x.device())
        });

        let gru_out = self.gru.forward(x, Some(h));
        let gru_out = self.dropout.forward(gru_out);
        let gru_out = self.output_projection.forward(gru_out);
        self.activation.forward(gru_out)
    }

    /// Runs the sequence through the model and returns only the last time-step output.
    ///
    /// Useful for sequence-to-one tasks such as next-step forecasting. Accepts
    /// an optional initial hidden state; `None` zero-initialises it.
    ///
    /// Returns a tensor of shape `[batch_size, output_size]`.
    ///
    /// # Panics
    ///
    /// Panics if the sequence-length dimension of `x` is zero.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use burn::backend::NdArray;
    /// # use burn::nn::Initializer;
    /// # use burn::tensor::Tensor;
    /// # use burn_gru_time_series_forecasting::gru_model::CustomGruRnnConfig;
    /// # let device = Default::default();
    /// # let model = CustomGruRnnConfig::new(4, 16, 4, true, false, 0.0,
    /// #     Initializer::XavierNormal { gain: 1.0 }).init::<NdArray>(&device);
    /// let x: Tensor<NdArray, 3> = Tensor::zeros([2, 10, 4], &device);
    /// let pred = model.forward_last(x, None); // [2, 4]
    /// ```
    pub fn forward_last(
        &self,
        x: Tensor<B, 3, Float>,
        h: Option<Tensor<B, 2, Float>>,
    ) -> Tensor<B, 2, Float> {
        let output = self.forward(x, h);
        let batch_size = output.dims()[0];
        let seq_len = output.dims()[1];
        let output_size = output.dims()[2];

        output
            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..output_size])
            .reshape([batch_size, output_size])
    }
}

/// Two-layer stacked GRU model with per-layer dropout and a linear output projection.
///
/// Each GRU layer feeds into the next, allowing the model to learn hierarchical
/// temporal representations. Suitable for sequence-to-sequence or sequence-to-one
/// regression tasks.
///
/// # Examples
///
/// ```no_run
/// use burn::backend::NdArray;
/// use burn::nn::Initializer;
/// use burn_gru_time_series_forecasting::gru_model::{StackedGruRnn, StackedGruRnnConfig};
///
/// let device = Default::default();
/// let model: StackedGruRnn<NdArray> = StackedGruRnnConfig::new(
///     4, 16, 4, true, 0.1, Initializer::XavierNormal { gain: 1.0 },
/// )
/// .init(&device);
/// ```
#[derive(Module, Debug)]
pub struct StackedGruRnn<B: Backend> {
    /// First GRU layer; consumes raw input features.
    gru_layer1: Gru<B>,
    /// Dropout applied after the first GRU layer.
    dropout1: Dropout,
    /// Second GRU layer; consumes the first layer's hidden states.
    gru_layer2: Gru<B>,
    /// Dropout applied after the second GRU layer.
    dropout2: Dropout,
    /// Projects hidden states from `hidden_size` to `output_size`.
    output_projection: Linear<B>,
}

/// Configuration for [`StackedGruRnn`]
#[derive(Config, Debug)]
pub struct StackedGruRnnConfig {
    /// Number of input features per time step.
    input_size: usize,
    /// Dimensionality of each GRU layer's hidden state.
    hidden_size: usize,
    /// Number of output features per time step.
    output_size: usize,
    /// Whether to include bias terms in the GRU gates.
    bias: bool,
    /// Dropout probability applied after each GRU layer; `0.0` disables dropout.
    dropout: f64,
    /// Weight initializer for both GRU layers and the linear projection.
    initializer: Initializer,
}

impl StackedGruRnnConfig {
    /// Initialise a [`StackedGruRnn`] model from this configuration
    ///
    /// Both GRU layers and the output projection are initialised on `device` using
    /// the same `initializer`. The second GRU layer receives `hidden_size` as its
    /// input size because it consumes the output of the first layer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn::backend::NdArray;
    /// use burn::nn::Initializer;
    /// use burn_gru_time_series_forecasting::gru_model::{StackedGruRnn, StackedGruRnnConfig};
    ///
    /// let device = Default::default();
    /// let model: StackedGruRnn<NdArray> = StackedGruRnnConfig::new(
    ///     4, 16, 4, true, 0.1, Initializer::XavierNormal { gain: 1.0 },
    /// )
    /// .init(&device);
    /// ```
    pub fn init<B: Backend>(&self, device: &B::Device) -> StackedGruRnn<B> {
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
    /// Runs a full sequence through both GRU layers and returns all time-step outputs.
    ///
    /// Hidden states for each layer are zero-initialised at the start of every call.
    ///
    /// Returns a tensor of shape `[batch_size, sequence_length, output_size]`.
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
    /// let x: Tensor<NdArray, 3> = Tensor::zeros([2, 10, 4], &device);
    /// let out = model.forward(x); // [2, 10, 4]
    /// ```
    pub fn forward(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        let batch_size = x.dims()[0];

        let h1 = Tensor::zeros(
            Shape::new([batch_size, self.gru_layer1.d_hidden]),
            &x.device(),
        );
        let x = self.gru_layer1.forward(x, Some(h1));
        let x = self.dropout1.forward(x);

        let h2 = Tensor::zeros(
            Shape::new([batch_size, self.gru_layer2.d_hidden]),
            &x.device(),
        );
        let x = self.gru_layer2.forward(x, Some(h2));
        let x = self.dropout2.forward(x);

        self.output_projection.forward(x)
    }

    /// Runs the sequence through both GRU layers and returns only the last time-step output.
    ///
    /// Useful for sequence-to-one tasks such as next-step forecasting. Hidden states
    /// are zero-initialised at the start of every call.
    ///
    /// Returns a tensor of shape `[batch_size, output_size]`.
    ///
    /// # Panics
    ///
    /// Panics if the sequence-length dimension of `x` is zero.
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
    /// let x: Tensor<NdArray, 3> = Tensor::zeros([2, 10, 4], &device);
    /// let pred = model.forward_last(x); // [2, 4]
    /// ```
    pub fn forward_last(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 2, Float> {
        let output = self.forward(x);
        let batch_size = output.dims()[0];
        let seq_len = output.dims()[1];
        let output_size = output.dims()[2];

        output
            .slice([0..batch_size, (seq_len - 1)..seq_len, 0..output_size])
            .reshape([batch_size, output_size])
    }
}
