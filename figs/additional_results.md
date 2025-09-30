## Ranking (in-context memory) using the very long benchmark, the narrative containing 2000 chapters, 1M tokens, and 35 Q&As.

| Model   | Simple Recall (million tokens book) | Chronological Awareness (million tokens book) |
|:---------------------------:|:-------------------:|:-------------------:|
gemini-2.5-pro          | 0.654  | 0.320

## Ranking (in-context memory) using the default long benchmark (Synaptic Echoes), the narrative containing 200 chapters, 100k tokens, and 686 Q&As. Those results are shown in the main README.md

| Model | Simple Recall | Chronological Awareness |
|:---------------------------:|:-------------------:|:-------------------:|
gemini-2.5-pro | 0.968 | 0.796
gemini-2.5-flash | 0.960 | 0.817
gpt-5 | 0.942 | 0.804
gpt-5-mini | 0.830 | 0.442
claude-sonnet-4 | 0.790     | 0.326
grok-4-fast-reasoning | 0.726    | 0.281
gemini-2-pro | 0.708 | 0.290
gemini-2-flash-thinking | 0.708 | 0.288
gpt-4o | 0.670 | 0.204
grok-4-fast-non-reasoning | 0.602 | 0.122
deepseek-v3 | 0.600	 | 0.103
gemini-2-flash | 0.596 | 0.173
deepseek-r1 |	0.572 | 0.147
llama-3.1-405b | 0.504 | 0.129 
gpt-4o-mini | 0.492 | 0.077
claude-3-haiku | 0.470 | 0.109 
claude-3-5-sonnet | 0.470 | 0.090
o3-mini | 0.424 | 0.044
o1 | 0.384 | 0.052
gpt-4.1-nano | 0.356 | 0.090
o1-mini | 0.300 | 0.033

## Ranking (in-context memory) using the short benchmark, the narrative containing 20 chapters, 10k tokens, and 456 Q&As.

| Model | Simple Recall (short book) | Chronological Awareness (short book) |
|:---------------------------:|:-------------------:|:-------------------:|
deepseek-r1 |	0.988 | 0.964
gemini-2.5-pro | 0.982 |0.948
grok-4-fast-reasoning | 0.982 | 0.932
gemini-2.5-flash |0.980 | 0.916
gpt-5 | 0.978 | 0.948
o1 | 0.978 | 0.948
claude-sonnet-4 | 0.972 | 0.657
gemini-2-flash-thinking | 0.962 | 0.967
gpt-5-mini | 0.962 | 0.948
gemini-2-pro | 0.950 | 0.186
o3-mini | 0.945 | 0.809
o1-mini | 0.935 | 0.809
grok-4-fast-non-reasoning | 0.930 | 0.512
gpt-5-nano | 0.920 | 0.764
gemini-2-flash | 0.915 | 0.163
deepseek-v3 | 0.910 | 0.481
gpt-4o | 0.908 | 0.182
llama-3.1-405b | 0.895 | 0.297
claude-3-5-sonnet | 0.845 | 0.329
gpt-4o-mini | 0.803 | 0.256
claude-3-haiku | 0.698 | 0.185
gpt-4.1-nano | 0.695 | 0.287