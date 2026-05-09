# Shape Flow Reference

## Symbols
- B: batch size
- S: sequence length
- H: backbone hidden size
- L: logic dimension
- G: number of gates
- N: number of backbone layers

## Backbone Hidden States
- hidden_states from backbone with output_hidden_states=True
- Typical length: N+1 (embedding output + N layer outputs)
- Use per-layer states for logic stream: hidden_states[1:]
- Each layer tensor shape: [B,S,H]

## Projection
For each layer i in 1..N:
- token_logic_i = LogicProjection(hidden_i)
- Shape: [B,S,L]

## One Logic Layer
Inputs:
- token_logic: [B,S,L]
- logic_state: [B,L]

Steps:
1. routing_weights = RoutingModule(token_logic)
   - Shape: [B,S,G]

2. gate_inputs = einsum("bsg,bsl->bgl", routing_weights, token_logic)
   - Shape: [B,G,L]

3. gate_values = sigmoid(Linear(L->1)(gate_inputs)).squeeze(-1)
   - Shape: [B,G]

4. gate_outputs = compose_gate_outputs(gate_values)
   - Shape: [B,G]

5. logic_delta = Linear(G->L)(gate_outputs)
   - Shape: [B,L]

6. new_logic_state = LayerNorm(logic_state + logic_delta)
   - Shape: [B,L]

Output:
- new_logic_state: [B,L]
- routing_weights: [B,S,G]

## Logic Stream Across N Layers
- Initialize logic_state = zeros([B,L])
- Iterate through N projected layer tensors
- Return:
  - final_logic_state: [B,L]
  - routing_history: list of N tensors, each [B,S,G]

## Fusion
Let:
- h_llm = final backbone layer output, shape [B,S,H]
- logic_bc = broadcast(final_logic_state), shape [B,S,L]

Steps:
1. combined = concat([h_llm, logic_bc], dim=-1)
   - Shape: [B,S,H+L]
2. correction = FusionMLP(combined)
   - Shape: [B,S,H]
3. fused = h_llm + alpha * correction
   - Shape: [B,S,H]

## Task Head
- pooled = fused[:,0,:] -> [B,H]
- logits = Linear(H->num_labels)(pooled) -> [B,num_labels]

## Baseline Path
Without logic stream:
- pooled backbone hidden -> task head
- logits shape remains [B,num_labels]
