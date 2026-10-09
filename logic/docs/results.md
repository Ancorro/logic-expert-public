
## First real test: [Part 9]:
Lora and non-Lora Logic:
Lora has higher routing_entropy and it does not reduce much at all
non-Lora reduces over training
grad_norm farlower in Lora, and stabilize,
non-Lora has high grad_norm that does not stabilize


## Gate VS no-gate

Gate:
horiz

Runtime device : cuda:0
Runtime dtype  : torch.float16
Causal pooling : True
Total params      : 7,515,199,779
Logic/head params : 10,271,011  (0.1%)


No gate(s)?:
== Parameter Report ==
Model class            : LogicLlamaModel
Total params           : 7,515,199,779
Trainable params       : 10,275,107
Added path params      : 10,266,913  (0.14%)
Added path trainable   : 10,266,913  (99.92% of trainable)
Added modules: logic_projection, logic_stream, fusion, pre_head_norm
Baseline-like estimate : 7,504,932,866  (backbone + task head)
