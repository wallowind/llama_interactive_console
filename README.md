### Original repository [and Readme] at https://github.com/facebookresearch/llama
## Llama **2** Interactive Console
Unlike the previous version [branch **llama_v1** in current repository], where I tried to allow multi-gpu inference, this version is the exact opposite and is designed to limit GPU requirements (especially in terms of VRAM) for Llama models as much as possible.
I couldn't do much about it, but the functioning of dynamically loading weights from RAM to VRAM on demand is probably worth sharing.
## What was done:
1) Removed all `fairscale` related stuff from model and rewrote code in pure `torch`
2) Now there is only two hard requiremets: `torch` (tested and worked in versions *1.8* and *1.13*) and `sentencepiece` (for tokenizer)
3) Added optional inference in *8bit* (requires `bitsandbytes`)
4) Added full cpu inference (slow, for proper implementation check project `llama.cpp`)
5) Added mixed cpu/gpu inference (can be combined with *8bit*)
6) Rewritted `Llama.build` function to allow combining multiple checkpoints into one (tested and implemented only for the 13B model, but the logic can be extended to larger models)
7) Added `Mirostat` (https://github.com/basusourya/mirostat) to the sampling methods
8) Added full caching mechanism *(in original Llama it was limited to single response and recalculated kv_cache for all previous dialogue turns)*
9) Added a *very basic* console interface for chatting *(definitely should have done more, given the repository name)*

Lots of implementation changes from the source code *(probably shouldn't have dumped everything into `generate`)*.

## Some results
Tests was done on two machines (results for the last one in the last column with an asterisk):
* Ryzen 5900X (12 cores)|128Gb RAM|GeForce RTX 3090 Ti|Ubuntu 23.04|PyTorch 1.13 (CUDA 11.7)
* Ryzen 1700X (8 cores)|64Gb RAM|GeForce GTX 1080 Ti|Ubuntu 16.04|PyTorch 1.8 (CUDA 10.1)

| Tokens/s (VRAM)   | 7B    | 13B   | 7B*   |
|:---:  |:---:  |:---:  |:---:  |
| **GPU**   | 45.38 (13.5)  | OOM   | OOM   |
| **CPU**   | 0.99 (26*)    |  0.55 (51*)   | 0.67 (25*)    |
| **8bit**  | 11.25 (8.3)   | 9.28 (14.6)   | -     |
| **Off_ALL**   | 1.81 (1.8)    | 0.93 (2.5)    | 0.74 (2.1)    |
| **Off_FF**    | 5.06 (9.6)    | 2.59 (17.9)   | 2.54 (9.9)    |
| **Off_ATT**   | 2.66 (5.6)    | 1.37 (10.0)   | 1.14 (5.9)    |
##### Notes:
* Star in CPU row means RAM
* bitsandbytes does not support old CUDA
* inference speed with offloading depends mainly on the speed of transferring weights from CPU to GPU *(faster PCI-E will give a speed boost and vice versa)*
* output speed with offloading FeedForward layers is faster because they are larger *(more weights on GPU → less slowdown in transfer)*

## Comand examples
- `python3 app.py /data/LLaMA2/llama-2-7b-chat/ /data/LLaMA2/tokenizer.model` — 7B model fully on GPU (**GPU** in table)
- `python3 app.py /data/LLaMA2/llama-2-7b-chat/ /data/LLaMA2/tokenizer.model --use_8bit` — model converted to 8bit via `bitsandbytes` on GPU (**8bit** in table)
- `python3 app.py /data/LLaMA2/llama-2-7b-chat/ /data/LLaMA2/tokenizer.model --use_offload --f16` — most weights on CPU with dynamic loading on GPU (**Off_ALL**)
- `python3 app.py /data/LLaMA2/llama-2-7b-chat/ /data/LLaMA2/tokenizer.model --use_offload --offload_exclusion [w1,w2,w3] --f16` — FeedForward layers on GPU and Attention on CPU with dynamic loadig, all weights in float16 (**Off_FF**)
