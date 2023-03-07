# Original repository [and Readme] at https://github.com/facebookresearch/llama

# Changes
- New *example.py* allows to use console for interactive prompting. Supports multiple gpu (tested with 13b model on two RTX3090)
Supports multiple gpu's (tested with 13b model on two RTX3090).
- Modified **llama/generate.py** to support the above functionality
- Batch size set to 1 (equivalent to no batches at all)
- The rest of the code is left unchanged
