# Residual Connections (Full ASCII Diagram)

This diagram maps residual paths across:
- Backbone transformer blocks (LLaMA-style)
- Layer-aligned logic stream updates
- Final fusion residual correction

```text
Residual Connections Map (LogicLlamaModel)

INPUT
  |
  v
Backbone Transformer (N blocks, standard residuals inside each block)
  For block i:
      x_i --------------------+------------------------------+
                              |                              |
                              v                              |
                         Self-Attn                           |
                              |                              |
                              +----> (+) ----> y_i ----------+
                                        ^
                                        |  (skip from x_i)

      y_i --------------------+------------------------------+
                              |                              |
                              v                              |
                             MLP                             |
                              |                              |
                              +----> (+) ----> x_{i+1} ------+
                                        ^
                                        |  (skip from y_i)

(collect hidden state from each block for logic path)


LOGIC STREAM (layer-aligned over projected backbone states)
  init: logic_state_0 = 0

  For logic layer k:
      logic_state_k --------------------+------------------------------+
                                        |                              |
      projected_state_k -> routing/gates -> logic_delta_k             |
                                        |                              |
                                        +----> (+) ----> z_k ----------+
                                                  ^
                                                  |  (skip from logic_state_k)

      z_k -> LayerNorm -> logic_state_{k+1}


FINAL FUSION RESIDUAL
  llm_hidden_final ---------------------+------------------------------+
                                        |                              |
  [llm_hidden_final || logic_state_T] -> MLP -> correction            |
                                        |                              |
                                        +----> (+) ----> fused_hidden -+
                                                  ^
                                                  |  (skip from llm_hidden_final)
                                          with scale alpha:
                                          fused_hidden = llm_hidden_final + alpha * correction
```

## Notes

- Backbone residual pattern follows pre-norm decoder blocks:
  - h = x + Attention(RMSNorm(x))
  - y = h + MLP(RMSNorm(h))
- Logic stream residual update occurs at every logic layer:
  - logic_state_{k+1} = LayerNorm(logic_state_k + logic_delta_k)
- Fusion is residual by design with learnable scalar alpha.
