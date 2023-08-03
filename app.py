import time
import argparse
from llama.generation import Dialog, Message, Llama


DEFAULT_SYSTEM_PROMPT = str(
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature. "
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)

# Llama still overly polite, even with this prompt (but this one is shorter and saves computation/VRAM)
DEFAULT_SYSTEM_PROMPT_SIMPLIFIED = str(
    "You are a helpful and honest assistant. If you don't know the answer to a question, please don't share false information. "
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct."
)

HELP_MSG = """
There is following commands You can use for tuning Llama response:
    q — Stop dialog and quit.
    e — Erase current dialog history and starts a new one.
    p — Sets `top_p` parameter to limit unlikely tokens (only works with temperature > 0).
    t — Sets `temperature` parameter. Higher values gives more variations in model answers,
        but could lead to less coherent and wholesome generation.
        NOTE: the effect of temperature grows non-linearly and is very noticeable for values greater than 2.
    l — Sets maximum length of reply to model in tokens. May cause abrupt ending of phrases,
        but prevents too wordy answers.
    h — Prints this help message.
    d — Turn on/off `debug` mode. Shows log_probs of generated phrases, number of tokens and time of generation.
    s — Allows You to set `SystemMessage` to model. It may influence overall generation
        direction, style of writing and so on.
    r — Reset all settings to defaults (starts new Dialog).
    b — Prints progress bar for current generation (may be usefull for slow cpu-inference).
    c — Turn on/off `full` cache. It may be usefull for speeding up slow inference (mostly for `load_on_cpu` option).
    m — Specifies the target value for `Mirostat`: a sampling method that can produce interesting generation results.
        The target value *kind of* reflects the desired perplexity of the text.
        Empirically better results yields values between 3 and 6.
        The text becomes more expressive as the value increases, but may lose relevance and stray off topic.
        A lower value will produce results reminiscent of the answers from the original Llama.
        There is one more setting that will be set when mirostat is turned on: the number of most likely tokens
        for the target surprise setting. Choose as for `top_k` settings (i.e. 20 to 100), or just leave it blank
        (enter nothing and press Enter) for automatic setting.
        Note: the implementation is based on `llama.cpp` and the official version from the `Mirostat` authors
Note: If Llama has not finished the answer, you can press Enter (enter a blank line) to let her finish.
"""


def parse_args():
    allowed_keys = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "attention_norm", "ffn_norm"]
    parser = argparse.ArgumentParser(description='Entry point for launch Llama.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('ckpt_dir', type=str,
                        help="Path to directory with Llama checkpoint.\nExample: /data/LLaMA2/llama-2-7b-chat/")
    parser.add_argument('tokenizer_path', type=str,
                        help="Path to `tokenizer.model` file.\nExample: /data/LLaMA2/tokenizer.model")
    parser.add_argument('--use_8bit', required=False, default=False, action="store_true",
                        help="Load Llama in 8bit.\nNOTE: requires `bitsandbytes` library installed.")
    parser.add_argument('--use_offload', required=False, default=False, action="store_true",
                        help="Load Llama on GPU with offloading some weights on CPU.\n"
                        "NOTE: use `offload_exclusion` option to tune this settings.")
    parser.add_argument('--offload_exclusion', required=False, default=[],
                        dest="exclude_from_offload", nargs="*", action="extend", metavar="key",
                        # choices=["wq", "wk", "wv", "wo", "w1", "w2", "w3", "attention_norm", "ffn_norm"],
                        help="Exclude layers from cpu offloading for all TransformerBlock.\n"
                        "Keys: {wq, wk, wv, wo, w1, w2, w3, attention_norm, ffn_norm}")
    parser.add_argument('--cpu', required=False, default=False, action="store_true", dest="load_on_cpu",
                        help="Load Llama on CPU (should work even in system with no GPU).")
    parser.add_argument('--f16', required=False, default=False, action="store_true", dest="enforce_float16",
                        help="Enforces loading weights in float16.\n"
                        "NOTE: checkpoints are stored in bfloat16, but are loaded in float16 in original code.")
    parser.add_argument('--max_len', type=int, required=False, default=2048, dest="max_seq_len", metavar="",
                        help="Sets limit to maximum context length of Llama.\n"
                        "NOTE: default=2048 (from the original code), but Llama can certainly work with up to 4096.")
    parser.add_argument('--sys', type=str, required=False, default=DEFAULT_SYSTEM_PROMPT, metavar="",
                        help="Sets `system_message` for Llama.\n"
                        "NOTE: by default, a slightly abbreviated version of the message from the original code is used.")
    result = parser.parse_args()
    keys = []
    for item in result.exclude_from_offload:
        keys.extend([k for k in allowed_keys if item.find(k) != -1])
    result.exclude_from_offload = keys
    return result


def main():
    kwargs = vars(parse_args())
    sys_msg = kwargs.pop('sys')
    llama = Llama.build(**kwargs)
    # Defaults
    dialog = None
    temperature = 0.0
    max_gen_len = 128
    top_p = 1.0
    mir_t = 0.0
    mir_l = 0
    logprobs = False
    show_bar = False
    use_cache = False
    print(f"*** SYSTEM_MSG:\nCurrent settings:\n\ttop_p={top_p}\n\tmax_len={max_gen_len}\n\ttemperature={temperature}"
          f"\n\tMirostat sampling not in use\n\tDebug mode is off\n\tProgress bar disabled\n"
          "Enter 'h' to see help message or 'q' to quit. ***")
    while True:
        try:
            msg = input("User:> ")
        # I have no idea why if happens sometimes...
        except UnicodeDecodeError:
            print("UnicodeDecodeError. Just enter (or copy/paste) your current message again.")
            continue
        if msg == "q":
            break
        elif msg == "e":
            dialog = None
            print("*** SYSTEM: New dialog starts. ***")
        elif msg == "p":
            top_p = float(input("*SYSTEM_INPUT: Top_p (in float): "))
        elif msg == "t":
            temperature = float(input("*SYSTEM_INPUT: Temperature (in float): "))
        elif msg == "l":
            max_gen_len = int(input("*SYSTEM_INPUT: Max_gen_len (in int): "))
        elif msg == "h":
            print(HELP_MSG)
        elif msg == "d":
            logprobs = not logprobs
            print(f"*** SYSTEM_MSG: Debug mode is {'on' if logprobs else 'off'}. ***")
        elif msg == "s":
            new_msg = input(f"*SYSTEM_INPUT: Input `SYSTEM_PROMPT` for Llama (left empty to reset to DEFAULT): ")
            sys_msg = DEFAULT_SYSTEM_PROMPT_SIMPLIFIED if len(new_msg) == 0 else new_msg
            dialog = None
        elif msg == "r":
            temperature = 0.0
            max_gen_len = 128
            top_p = 1.0
            mir_t = 0.0
            mir_l = 0
            logprobs = False
            show_bar = False
            use_cache = False
            print(f"*** SYSTEM_MSG: Setting reset to default values. ***")
        elif msg == "b":
            show_bar = not show_bar
            print(f"*** SYSTEM_MSG: Progress bar {'enabled' if show_bar else 'disabled'}. ***")
        elif msg == "c":
            use_cache = not use_cache
            print(f"*** SYSTEM_MSG: Full cache is {'enabled' if use_cache else 'disabled'}. ***")
        elif msg == "m":
            mir_t = float(input("*SYSTEM_INPUT: Mirostat target value (in float): "))
            mir_l = input("*SYSTEM_INPUT: Mirostat limit (in int or left empty): ")
            mir_l = int(mir_l) if len(mir_l) else 0
        else:
            if dialog is None:
                dialog = Dialog(msg, system=sys_msg)
            else:
                dialog += Message("user", msg)
            if logprobs:
                start = time.time()
            output = llama.chat_completion(
                dialog=dialog,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                mir_t=mir_t,
                mir_l=mir_l,
                logprobs=logprobs,
                show_bar=show_bar,
                use_cache=use_cache,
            )
            if output is None:
                print(f"*** SYSTEM_MSG: Llama has not generated a single token (give her some context...) ***")
                dialog.pop(-1)
                continue
            dialog += output["generation"]
            # Sets cursor on previous line (erase empty user's message)
            # prefix = f"\033[F" if len(msg) == 0 else ""
            prefix = ""
            if logprobs:
                print(f"Bot:> {prefix}{dialog[-1].content}")
                print(f"*** SYSTEM_MSG: Generated {len(output['tokens'])} tokens with perplexity "
                      f"{output['perplexity']} in {(time.time() - start):.2f} s. "
                      f"[A full history consists of {output['full_tokens']} tokens.] ***")
            else:
                print(f"Bot:> {prefix}{dialog[-1].content}")


if __name__ == '__main__':
    main()
