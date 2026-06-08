# GRU RNN Architecture Diagrams

## 1. The Problem: Shape Mismatch

```
┌─────────────────────────────────────────────────────────────┐
│                     GRU Forward Pass                        │
└─────────────────────────────────────────────────────────────┘

Input (x)                  Hidden State (h)
[1, 4, 3]                  [1, 8]
   │                          │
   │                          │
   └──────────┬───────────────┘
              │
              ▼
    ┌──────────────────┐
    │   GRU Layer      │
    │ (input=3,        │
    │  hidden=8)       │
    └──────────────────┘
              │
              ▼
        y_hat Output
        [1, 4, 8]
              │
              │
              ▼
        ❌ MISMATCH!
              │
              ▼
         Target (y)
         [1, 4, 3]

   Hidden Size = 8 ≠ Output Size = 3
```

---

## 2. The Solution: Linear Output Projection

```
┌─────────────────────────────────────────────────────────────┐
│              Complete RNN Architecture                       │
└─────────────────────────────────────────────────────────────┘

Input (x)                  Hidden State (h)
[1, 4, 3]                  [1, 8]
   │                          │
   │                          │
   └──────────┬───────────────┘
              │
              ▼
    ┌──────────────────┐
    │   GRU Layer      │
    │ (input=3,        │
    │  hidden=8)       │
    └──────────────────┘
              │
              ▼
        [1, 4, 8]
              │
              ▼
    ┌──────────────────┐
    │  Reshape 3D→2D   │
    │  [1×4, 8]        │
    └──────────────────┘
              │
              ▼
         [4, 8]
              │
              ▼
    ┌──────────────────┐
    │  Linear Layer    │  ⚠️ REQUIRED
    │  (8 → 3)         │
    └──────────────────┘
              │
              ▼
         [4, 3]
              │
              ▼
    ┌──────────────────┐
    │  Reshape 2D→3D   │
    │  [1, 4, 3]       │
    └──────────────────┘
              │
              ▼
        y_hat_final
        [1, 4, 3]
              │
              │
              ▼
        ✅ MATCH!
              │
              ▼
         Target (y)
         [1, 4, 3]
```

---

## 3. CustomGruRnn Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CustomGruRnn Model                         │
│  Input: [batch, seq, input_size]                            │
│  Output: [batch, seq, output_size]                          │
└─────────────────────────────────────────────────────────────┘

        Input Tensor
    [batch, seq, input_size]
              │
              ▼
    ┌──────────────────┐
    │  Hidden State    │
    │  Initialization  │
    │  (zeros)         │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │   GRU Layer      │  🔄 Recurrent Processing
    │   (input_size    │
    │    → hidden_size)│
    └──────────────────┘
              │
              ▼
    [batch, seq, hidden_size]
              │
              ▼
    ┌──────────────────┐
    │  Dropout Layer   │  🎯 Regularization
    │  (p=0.2)         │
    └──────────────────┘
              │
              ▼
    [batch, seq, hidden_size]
              │
              ▼
    ┌──────────────────┐
    │  Reshape         │
    │  [batch×seq,     │
    │   hidden_size]   │
    └──────────────────┘
              │
              ▼
    [batch×seq, hidden_size]
              │
              ▼
    ┌──────────────────┐
    │  Linear Layer    │  ⚠️ Output Projection
    │  (hidden_size    │
    │   → output_size) │
    └──────────────────┘
              │
              ▼
    [batch×seq, output_size]
              │
              ▼
    ┌──────────────────┐
    │  Reshape         │
    │  [batch, seq,    │
    │   output_size]   │
    └──────────────────┘
              │
              ▼
        Output Tensor
    [batch, seq, output_size]
```

---

## 4. StackedGruRnn Architecture (Multi-Layer)

```
┌─────────────────────────────────────────────────────────────┐
│                 StackedGruRnn Model (2-Layer)                │
│  Input: [batch, seq, input_size]                            │
│  Output: [batch, seq, output_size]                          │
└─────────────────────────────────────────────────────────────┘

        Input Tensor
    [batch, seq, input_size]
              │
              ▼
    ┌──────────────────┐
    │  GRU Layer 1     │  🔄 First Level
    │  (input_size     │     Abstract Features
    │   → hidden_size) │
    └──────────────────┘
              │
              ▼
    [batch, seq, hidden_size]
              │
              ▼
    ┌──────────────────┐
    │  Dropout 1       │
    │  (p=0.3)         │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  GRU Layer 2     │  🔄 Second Level
    │  (hidden_size    │     Higher-Order Features
    │   → hidden_size) │
    └──────────────────┘
              │
              ▼
    [batch, seq, hidden_size]
              │
              ▼
    ┌──────────────────┐
    │  Dropout 2       │
    │  (p=0.3)         │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  Reshape & Linear│  ⚠️ Output Projection
    │  (hidden_size    │
    │   → output_size) │
    └──────────────────┘
              │
              ▼
        Output Tensor
    [batch, seq, output_size]
```

---

## 5. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Component Dependencies                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐
│  Input Data │
│  [B, S, I]  │
└──────┬──────┘
       │
       │ ┌─────────────────┐
       │ │ Hidden State    │
       │ │ Initialization  │
       │ │ (Optional)      │
       │ └────────┬────────┘
       │          │
       ▼          ▼
   ┌───────────────────┐
   │   GRU Layer       │◄───┐
   │                   │    │
   │  - Input Gates    │    │ Recurrence
   │  - Reset Gates    │    │ (Temporal)
   │  - Update Gates   │    │
   │  - Candidate      │────┘
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │   Dropout         │  ⚡ Optional but Recommended
   │   (Regularization)│
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │  Linear Projection│  ⚠️ REQUIRED
   │  (Shape Transform)│
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │   Output          │
   │   [B, S, O]       │
   └───────────────────┘
             │
             ▼
   ┌───────────────────┐
   │   Loss Function   │
   │   (MSE, CE, etc.) │
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │   Optimizer       │
   │   (Adam, SGD)     │
   └───────────────────┘
```

---

## 6. Shape Transformation Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Detailed Shape Transformation                   │
└─────────────────────────────────────────────────────────────┘

Step 1: Input
┌──────────────────────┐
│  Batch Size:    1    │
│  Sequence Len:  4    │
│  Input Size:    3    │
│  Shape: [1, 4, 3]    │
└──────────┬───────────┘
           │
           ▼
Step 2: GRU Processing
┌──────────────────────┐
│  Batch Size:    1    │
│  Sequence Len:  4    │
│  Hidden Size:   8    │
│  Shape: [1, 4, 8]    │
└──────────┬───────────┘
           │
           ▼
Step 3: Flatten Sequence
┌──────────────────────┐
│  Batch×Seq:     4    │  (1 × 4 = 4)
│  Hidden Size:   8    │
│  Shape: [4, 8]       │
└──────────┬───────────┘
           │
           ▼
Step 4: Linear Projection
┌──────────────────────┐
│  Batch×Seq:     4    │
│  Output Size:   3    │  ⚠️ Transformed!
│  Shape: [4, 3]       │
└──────────┬───────────┘
           │
           ▼
Step 5: Reshape to 3D
┌──────────────────────┐
│  Batch Size:    1    │
│  Sequence Len:  4    │
│  Output Size:   3    │
│  Shape: [1, 4, 3]    │  ✅ Matches Target!
└──────────────────────┘
```

---

## 7. Training Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop Flow                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Dataset        │
│  └─ Train       │
│  └─ Validation  │
└────────┬────────┘
         │
    FOR EACH EPOCH
         │
    ┌────▼────────────────────────────────────┐
    │  Data Loader (Batching)                 │
    └────┬────────────────────────────────────┘
         │
    FOR EACH BATCH
         │
    ┌────▼────────────────────────────────────┐
    │  1. Forward Pass                        │
    │     model.forward(x_batch)              │
    │     → predictions [B, S, O]             │
    └────┬────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  2. Compute Loss                        │
    │     loss_fn(predictions, targets)       │
    │     → scalar loss value                 │
    └────┬────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  3. Backward Pass                       │
    │     loss.backward()                     │
    │     → compute gradients                 │
    └────┬────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  4. Update Weights                      │
    │     optimizer.step()                    │
    │     → apply gradients                   │
    └────┬────────────────────────────────────┘
         │
         │ END BATCH LOOP
         │
    ┌────▼────────────────────────────────────┐
    │  5. Validation                          │
    │     - Evaluate on validation set        │
    │     - Compute metrics                   │
    │     - Check for overfitting             │
    └────┬────────────────────────────────────┘
         │
         │ END EPOCH LOOP
         │
    ┌────▼────────────────────────────────────┐
    │  6. Final Model                         │
    │     - Save best checkpoint              │
    │     - Evaluate on test set              │
    └─────────────────────────────────────────┘
```

---

## 8. Use Case Architectures

### A. Sequence-to-Sequence (Time Series Forecasting)

```
Input: Historical prices [batch, 100 timesteps, 1 feature]
Output: Future prices   [batch, 100 timesteps, 1 prediction]

    [B, 100, 1]
         │
         ▼
    ┌─────────┐
    │   GRU   │
    └────┬────┘
         │
         ▼
    [B, 100, H]
         │
         ▼
    ┌─────────┐
    │ Linear  │
    └────┬────┘
         │
         ▼
    [B, 100, 1]  ← All timesteps predicted
```

### B. Sequence-to-One (Classification)

```
Input: Text sequence    [batch, seq_len, embedding_dim]
Output: Class logits    [batch, num_classes]

    [B, S, E]
         │
         ▼
    ┌─────────┐
    │   GRU   │
    └────┬────┘
         │
         ▼
    [B, S, H]
         │
         ▼
    ┌─────────┐
    │  Last   │  ← Extract final timestep
    │Timestep │
    └────┬────┘
         │
         ▼
      [B, H]
         │
         ▼
    ┌─────────┐
    │ Linear  │
    └────┬────┘
         │
         ▼
     [B, C]  ← Class predictions
```

### C. Many-to-Many (Translation)

```
Encoder-Decoder Architecture

Source: [B, S_src, E]  →  Target: [B, S_tgt, E]

    [B, S_src, E]
         │
         ▼
    ┌──────────┐
    │ Encoder  │
    │   GRU    │
    └────┬─────┘
         │
         ▼
   Context State
     [B, H]
         │
         ├────────────────┐
         │                │
         ▼                ▼
    ┌──────────┐    [B, S_tgt, E]
    │          │         │
    │ Decoder  │◄────────┘
    │   GRU    │
    └────┬─────┘
         │
         ▼
    [B, S_tgt, H]
         │
         ▼
    ┌──────────┐
    │  Linear  │
    └────┬─────┘
         │
         ▼
    [B, S_tgt, V]  ← Vocabulary predictions
```

---

## Legend

```
┌─────────────────────────────────────────────────────┐
│  Symbol      Meaning                                │
├─────────────────────────────────────────────────────┤
│  [B, S, I]   Batch, Sequence, Input dimensions      │
│  [B, S, H]   Batch, Sequence, Hidden dimensions     │
│  [B, S, O]   Batch, Sequence, Output dimensions     │
│  ⚠️           Required component                     │
│  🎯           Highly recommended                     │
│  ⚡           Optional but beneficial                │
│  🔄           Recurrent connection                   │
│  ✅           Success / Match                        │
│  ❌           Error / Mismatch                       │
└─────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Linear output layer is REQUIRED** to transform hidden states to target dimensions
2. **Dropout is HIGHLY RECOMMENDED** for regularization
3. **Multi-layer stacking** increases model capacity
4. **Proper shape management** is crucial at every step
5. **Different use cases** require different architectural patterns

---

For implementation details, see:
- `main.rs` - Primary example
- `ARCHITECTURE.md` - Comprehensive guide
