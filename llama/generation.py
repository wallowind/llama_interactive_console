# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import re
import json
import time
import tqdm
import warnings
from pathlib import Path
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

# from llama.model import ModelArgs, Transformer
from .model import ModelArgs, Transformer, Attention, TransformerBlock
from .tokenizer import Tokenizer


class Message(object):
    ROLES = ["user", "assistant"]
    def __init__(self, role: str, content: str):
        assert role in self.ROLES, f"Role must be from {self.ROLES} list."
        self.role = role
        self.content = content
        # Hack to preserve original text without <INST> tokens
        self.__text = content

    def get_text(self):
        return self.__text

    def __repr__(self):
        return f"<{self.role}: {self.__text[:25]}...>\n"


class Dialog(list):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    def __init__(self, msg: str, system: str):
        super().__init__()
        self.system_msg = system
        self.ensure_system(msg, system)

    def __add__(self, other):
        if isinstance(other, Message):
            self.append(other)
            return self
        else:
            return super().__add__(other)

    def __iadd__(self, other):
        if isinstance(other, Message):
            return self.__add__(other)
        else:
            return super().__iadd__(other)

    def append(self, other):
        if isinstance(other, Message):
            if other.role == "user":
                other.content = f"{self.B_INST} {other.content.strip()} {self.E_INST}"
            elif other.role == "assistant":
                # need to keep spaces for continuation (strip is called in `encode_turns`)
                # other.content = f"{other.content.strip()}"
                other.content = f"{other.content}"
            else:
                raise ValueError(f"Message can only be from user or assistant. Got role: {other.role}")
        return super().append(other)

    def ensure_system(self, user_msg: str, system_msg: str):
        assert len(self) == 0, "ensure_system must be called at initialization of Dialog"
        msg = self.B_SYS + system_msg + self.E_SYS + user_msg
        self.append(Message("user", msg))
        self.just_initialized = True

    def endcode_turns(self, tokenizer: Tokenizer, continuation: bool = False):
        assert all([msgs.role == "user" for msgs in self[::2]]) and all(
            [msgs.role == "assistant" for msgs in self[1::2]]), (
            "model only supports 'user' and 'assistant' roles, "
            "starting with 'user' (including `system`), and alternating (u/a/u/a/u...)"
        )
        assert self[-1].role == "user", "Last message must be from user"
        # NOTE: spaces is important?
        tokens = sum([tokenizer.encode(f"{p.content} {a.content.strip()} ", True, True)
                     for p, a in zip(self[::2], self[1::2])], [])
        # The user's empty message is not added to the dialogue so that Llama can continue her reply
        if continuation:
            return tokens[:-1]  # -1 for dropping eos and -2 for final space?
        tokens += tokenizer.encode(self[-1].content, True, False)
        return tokens


@contextmanager
def empty_init():
    # Borrowed from transformers, but removed requirements for `meta` device,
    # which is not presented in torch 1.8
    old_register_parameter = torch.nn.Module.register_parameter

    def register_none_parameter(module, name, param):
        # Weights initialization is not needed
        # Of all layers in Llama (Embedding, RMSNorm and Linear)
        # only latter has this stupid reset in __init__
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters = lambda: None
        old_register_parameter(module, name, param)
        if param is not None:
            # param is either torch.nn.Parameter or bitsandbytes.nn.modules.Int8Params
            # both can be initialized with empty tensors and have only `requires_grad` in kwargs
            param_cls = type(module._parameters[name])
            module._parameters[name] = param_cls(torch.tensor([]), requires_grad=False)
    try:
        torch.nn.Module.register_parameter = register_none_parameter
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        pass


def model_loading(
    params: ModelArgs,
    state_dict: Dict[str, torch.Tensor],
    use_8bit: bool = False,
    use_offload: bool = False,
    load_on_cpu: bool = False,
    exclude_from_offload: Optional[List[str]] = None,
    force16f: bool = False,
):
    # All keys in TransformerBlock, available for replacing/offloading
    NAMES = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "attention_norm", "ffn_norm"]
    # Keys forbidden to replace/offload (input embedding, output linear and RMSNorm for output)
    # Note: there is no technical limitations for replacing/offloading those layers
    EXCLUDE = ["tok_embeddings", "output", "norm"]
    if use_8bit:
        import os
        # copy/pasted from text-generation-webui
        os.environ['BITSANDBYTES_NOWELCOME'] = '1'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import bitsandbytes as bnb

        def replace_8bit_linear(model, exclude=[], threshold=6.0):
            # copy/paste from transformers
            for name, module in model.named_children():
                if len(list(module.children())) > 0:
                    replace_8bit_linear(module, exclude, threshold)
                if isinstance(module, torch.nn.Linear) and name not in exclude:
                    with empty_init():
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=False,
                            has_fp16_weights=False,
                            threshold=threshold)

    # Model with all parameters settled to None (No RAM or VRAM consumption)
    with empty_init():
        model = Transformer(params)
    # -------------------------------------------------------------------------------- #
    # Simplest settings: just maps weights from state_dict to model (with cast to float32)
    # No GPU needed, but consumes alot of RAM (~28Gb for 7B and twice as much for 13B)
    # Inference is quite slow. If You really wants running Llama on cpu,
    # better looking on `llama.cpp` project
    # ***
    if load_on_cpu:
        for name, module in model.named_modules():
            weight = state_dict.pop(f"{name}.weight", None)
            if weight is None:
                continue
            module._parameters["weight"] = weight.to(torch.float32)
        return model
    # `freqs_cis` should be on gpu (in original code placement occurs in forward())
    model.freqs_cis = model.freqs_cis.to("cuda")
    # -------------------------------------------------------------------------------- #
    # Loads parts of model on GPU and lefts the rest on CPU
    # Dynamically loads data from CPU for calculations
    # Balanced VRAM/RAM usage (can be tuned to via `exclude_from_offload` parameter)
    # Inference speed varies from slow to very slow, depending on available pci-e lanes of GPU
    # Can be combined with 8bit to further reduce VRAM consumption (at the cost of slightly slower inference speed)
    # ***
    if use_offload:
        # This function finds module names for TransformerBlock in state_dict's keys
        # regex is't necessary here (all options could'be hardcoded)
        def parse_name(name: str):
            result = re.findall(r"(?:layers\.\d{1,}\.)(.+?(?=\.))*(?:\.)*(.+?(?=\.|$))", name)
            if len(result):
                return [v for v in result[0] if len(v)]
            return None
        if exclude_from_offload is None:
            exclude_from_offload = ["wq", "wk", "wv", "wo"]  # TODO: Check optimal exclusion.
        # Initialize `buffers` for offloaded weights
        # In `keys` will be placed links to TransformerBlock layers of model,
        # and `values` consists of dictionaries with items:
        # {`universal` gpu_layer module: weight of `actuall` module, which resides on cpu}
        keys, values = list(), list()
        # `universal` gpu_layer. All offloaded parameters of `actuall` layers are linked to gpu_layer
        # Example: [A1.wq = G.wq, A2.wq = G.wq, ...] (for all layers with offloaded wq)
        gpu_layer = TransformerBlock(params)  # Created on cpu...
        for name, module in gpu_layer.named_modules():
            # remove parameters, which is not offloaded
            if name.split(".")[-1] in exclude_from_offload:
                module._parameters["weight"].data = torch.tensor([])
        gpu_layer.cuda()  # and moved to gpu.
        if use_8bit:
            # This is `inverse` of `exclude_from_offload`, because `excluded` from
            # offload modules resides on GPU and should be converted to 8bit
            exclude = [v for v in NAMES if v not in exclude_from_offload]
            exclude.extend(EXCLUDE)
            # After this line all non-ofloaded layers are uninitialized 8bit
            replace_8bit_linear(model, exclude)
        for name, module in model.named_modules():
            # NOTE: `named_modules` is OrderedDict and goes from parent to children,
            # therefore the ordering of `cpu_layers` and `actual` layers is preserved
            if isinstance(module, TransformerBlock):
                keys.append(module)
                values.append(dict())
            weight = state_dict.pop(f"{name}.weight", None)
            if weight is None:
                continue
            # This option may be necessary for older GPU/CUDA without bfloat16 support
            # Also original code loads weights in float16 for unknown reason
            if force16f:
                weight = weight.half()
            if any([n == name for n in EXCLUDE]):
                # Note: `.data` is't necessary: tensors (instead of Parameters)
                # works just fine for inference
                module._parameters["weight"].data = weight.to("cuda")
                continue
            names = parse_name(name)
            assert names is not None, f"Parsing error for {name}"
            if names[-1] in exclude_from_offload:
                if use_8bit:
                    param_cls = type(module._parameters["weight"])
                    # Actuall convertation in 8bit happens in this line (note: weight must be float16)
                    module._parameters["weight"] = param_cls(weight.half(), requires_grad=False).to("cuda")
                else:
                    # Next try/except is because some weird bug somwhere in cuda (or torch?)
                    try:
                        # This should work and usually does, but somethimes...
                        module._parameters["weight"].data = weight.cuda()
                    # There is `RuntimeError: CUDA error: OS call failed or operation not supported on this OS`
                    except RuntimeError:
                        # print(name)  # uncomment to see what modules failed
                        # However, these two lines works... I think it has something to do with the
                        # `pin_memory` problems, but I don't know what to do
                        d = weight.cuda()
                        module._parameters["weight"].data = d
            else:
                # overcomplicated logic to find parameter in gpu_layer
                # basically unwrapped recurrent calls to module.named_children()
                # key_name = ".".join(names)
                gpu_module = gpu_layer
                while len(names):
                    gpu_module = gpu_module._modules[names.pop(0)]
                # put offloaded parameters on cpu
                # Note: pin_memory() is important for performance
                # Note 2: pin_memory can have weird limits (depended on OS in use?),
                # thats why there is try/except block
                try:
                    # memory allocated without problem and weight is pinned
                    values[-1][gpu_module] = weight.pin_memory()
                except RuntimeError:
                    # allocation error and weight is left unpinned
                    # everything works anyway, just slower...
                    values[-1][gpu_module] = weight
                # link `actual` module parameters to `universal` parameter of gpu_module
                module._parameters["weight"] = gpu_module._parameters["weight"]
        # mapping for all modules: {`actuall` module: {`universal` module: weight on cpu}}
        layers_mapping = {k: v for k, v in zip(keys, values)}
        # Hook for loading weight from cpu to `universal` module in gpu_layer
        def preload_hook(module, _):
            block = layers_mapping[module]
            for m, t in block.items():
                # Note: `non_blocking` slightly improves transfer speed and
                # does't cause any `race condition` due to synchronization,
                # implemented in torch for `default` cuda stream.
                m._parameters["weight"].data = t.cuda(non_blocking=True)
        # Registering pre_forward hook
        for m in model.layers:
            m.register_forward_pre_hook(preload_hook)
        return model
    # -------------------------------------------------------------------------------- #
    # Loads model in 8bit: almost halves VRAM usage (~8Gb for 7B and ~15Gb for 13B),
    # but incurs slowdown in 4–5 times. No RAM required.
    # ***
    if use_8bit:
        replace_8bit_linear(model, exclude=EXCLUDE)
        for name, module in model.named_modules():
            weight = state_dict.pop(f"{name}.weight", None)
            if weight is None:
                continue
            param_cls = type(module._parameters["weight"])
            module._parameters["weight"] = param_cls(weight.half(), requires_grad=False).to("cuda")
        return model
    # -------------------------------------------------------------------------------- #
    # Load full model on GPU: ~14Gb VRAM for 7B and OOM (on 3090) for 13B
    # Fastest (much faster than any other option) inference speed and no RAM reqirements
    # ***
    # After this line any tensor will be in float16 and automatically placed on gpu
    # NOTE: the actual weights are in bfloat16, but when loaded they are cast to float16,
    # not sure why, but this is a copy/paste from the original code
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # empty model is not needed here, just initialize Transformer on gpu
    model = Transformer(params)
    # And load weights from state_dict...
    # Note: `strict=False` was needed only because of "rope.freqs"
    model.load_state_dict(state_dict, strict=True)
    return model


class Llama:
    @ staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        use_8bit: bool = False,
        use_offload: bool = False,
        load_on_cpu: bool = False,
        # Note: `max_seq_len` in the configuration file is set to 2048, but the model can
        # handle a context twice as long (see `precompute_freqs_cis` in model.py or the article),
        # but it can be limited to a smaller value to reduce VRAM consumption
        max_seq_len: int = 2048,
        exclude_from_offload: Optional[List[str]] = None,
        enforce_float16: bool = False,
    ) -> "Llama":

        assert torch.cuda.device_count() <= 1, (
            "Only single GPU settings supported. Use standart Llama for multi-gpu support.",
            "For multi-gpu system use `export CUDA_VISIBLE_DEVICES=X` before launch the script."
        )

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"no checkpoint files found in {ckpt_dir}")
        elif len(checkpoints) == 1:
            state_dict = torch.load(checkpoints[0], map_location="cpu")
            state_dict.pop("rope.freqs")
        elif len(checkpoints) == 2:
            state_dict = Llama.collect_two_ckpts(*checkpoints)
        else:
            raise NotImplementedError(f"Only 7b and 13b models supported.")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(**params)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words  # Already in ModelArgs
        model_args.max_seq_len = max_seq_len
        model = model_loading(model_args, state_dict, use_8bit, use_offload,
                              load_on_cpu, exclude_from_offload, enforce_float16)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.show_bar = False

    def install_kv_cache_hooks(self, size: int, cache: Optional[dict] = None, idx: Optional[int] = None):
        # copy/pasted from whisper (and `slightly` rewritten)
        # Dynamically creates kv_cache with exact lenght, asked for generation
        # Reduces VRAM usage without much of a performance hit
        # Also allows for `full` caching of dialog history, while in original
        # code all previous turns are recalculated and caching only works in
        # generation of a reply from model (it is quite fast though)

        n_heads = self.model.params.n_heads
        dim = self.model.params.dim
        head_dim = dim // n_heads
        hooks = []

        if cache is not None:
            assert idx is not None, "Need position for old cache."
            cache = {k: torch.cat((v[:, :idx], torch.empty(size=(1, size, dim), dtype=v.dtype, device=v.device)), dim=1)
                     for k, v in cache.items()}
            pos = {k: idx for k in cache.keys()}
        else:
            cache, pos = {}, {}

        def save_to_cache(module, _, output):
            # `module` is `wk` or `wv` layer from Attention and `ouput` is the result of its forward() call
            if module not in cache:  # initialize cache with requested size
                # set initial offset
                new_pos = output.size(1)
                # allocate memory for cache
                # Note: memory allocation is't free, so this will incur small performance hit in comparison
                # to original code (where caches initialized with `max_seq_len` size during model creation)
                cache[module] = torch.empty(size=(1, size, dim), dtype=output.dtype, device=output.device)
                # save as-is, for the first forward(), when cache is empty
                cache[module][:1, :new_pos] = output
                # remember current offset in cache
                pos[module] = new_pos
            else:  # place new ouput into cache
                new_pos = pos[module] + output.size(1)
                # print(f"{pos[module]} <-> {new_pos} || {output.size()} <-> {cache[module].size()}")
                cache[module][:1, pos[module]:new_pos] = output
                pos[module] = new_pos
            # return cache with all previous and current output combined
            # Note: reshaping here is't necessary (it could be done in Attention's forward() instead)
            return cache[module][:, :new_pos].view(1, new_pos, n_heads, head_dim)

        # Attaches hooks to layers. Hook is called right after forward() call of wk/wv layers
        def install_hooks(layer: torch.nn.Module):
            if isinstance(layer, Attention):
                hooks.append(layer.wk.register_forward_hook(save_to_cache))
                hooks.append(layer.wv.register_forward_hook(save_to_cache))
        self.model.apply(install_hooks)
        return hooks, cache, pos

    @staticmethod
    def collect_two_ckpts(ckpt1: Path, ckpt2: Path):
        # Note: first load is very long, but consecutive loads much faster
        # The memory consumption is always very high (over 26Gb of RAM, but 32Gb should be enough)
        print("Merging checkpoints. It may take some time and consumes lots of RAM.")
        left = torch.load(ckpt1, map_location="cpu")
        right = torch.load(ckpt2, map_location="cpu")
        keys = list(left.keys())
        for k in keys:
            # pop's for controlling RAM consumption
            lv, rv = left.pop(k), right.pop(k)
            if "tok_embeddings" in k:
                right[k] = torch.cat((lv, rv), dim=1)
            # RMSNorm replicated on all GPUs
            elif "norm" in k:
                right[k] = rv
            # ColumnParallel ← splits among GPUs in first dimension
            elif k.split(".")[-2] in ("wq", "wk", "wv", "w1", "w3"):
                right[k] = torch.cat((lv, rv), dim=0)
            # RowParallel ← splits among GPUs in second dimension
            elif k.split(".")[-2] in ("wo", "w2"):
                right[k] = torch.cat((lv, rv), dim=1)
            # Last layer is ColumnParallel
            elif "output" in k:
                right[k] = torch.cat((lv, rv), dim=0)
            # Litter not used anywhere
            elif "rope" in k:
                continue
            else:
                raise KeyError(f"{k} is not supposed to be in llama's state_dict.")
        return right  # state_dict from combined checkpoints

    @torch.no_grad()  # @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        mir_t: float = 0.0,
        mir_l: int = 0,
        logprobs: bool = False,
        echo: bool = False,
        # turn on tqdm progress bar for generation
        show_bar: bool = False,
    ) -> Tuple[List[int], Optional[List[float]]]:

        pad_id = self.tokenizer.pad_id
        prompt_len = len(prompt_tokens)
        total_len = max_gen_len + prompt_len

        # Embedding always on correct device
        # Note: tokens will be on cpu only when Llama builded with `load_on_cpu`
        tokens = torch.full((1, total_len), pad_id, dtype=torch.long,
                            device=self.model.tok_embeddings.weight.device)
        tokens[0, :prompt_len] = torch.tensor(prompt_tokens).type_as(tokens)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = False

        if show_bar:
            bar = tqdm.tqdm(desc="Generating...", total=max_gen_len, leave=False)
        for cur_pos in range(prompt_len, total_len):
            logits = self.model.forward(tokens[:1, prev_pos:cur_pos])
            if mir_t > 0:  # Note: sampler does exist at this point (it was created in `chat_completion`)
                t = 1 if temperature == 0 else temperature
                if mir_l > 0:
                    next_token = self.mirostat(logits[:, -1].view(-1) / t, n_toks=mir_l)
                else:
                    next_token = self.mirostat(logits[:, -1].view(-1) / t)
            elif temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            # Note: logprobs update must be called after picking `next_token`
            if logprobs:
                token_logprobs[:, prev_pos:cur_pos] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1: cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached = next_token == self.tokenizer.eos_id
            prev_pos = cur_pos
            if eos_reached:
                break
            if show_bar:
                bar.update(1)
        if show_bar:
            bar.close()
        start = 0 if echo else prompt_len
        tokens = tokens.view(-1).tolist()[start:]
        # Not sure if this is correct `perplexity` calculation
        probs = token_logprobs.view(-1)[start:] if logprobs else None
        if self.tokenizer.eos_id in tokens:
            eos_idx = tokens.index(self.tokenizer.eos_id)
            tokens = tokens[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        return (tokens, probs)

    def text_completion(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ):  # Note: this function is largely left as it was in the original code
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        assert len(prompt_tokens) + max_gen_len < self.model.params.max_seq_len, "Too much context."
        gen_tokens, gen_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return {
                "generation": self.tokenizer.decode(gen_tokens),
                "tokens": gen_tokens,
                "logprobs": gen_logprobs,
            }
        return {"generation": self.tokenizer.decode(gen_tokens)}

    # utility function for preparing tokens for generation, taking into account the new caching systems and `Dialog`
    def manage_history(self, dialog: Dialog, max_gen_len: int, use_cache: bool = False):
        # flag for `continuation` previous reply from Llama (empty msg from user)
        no_msg = len(dialog[-1].get_text()) == 0
        # Sampler (if needed) is created in `chat_completion`
        sampler = getattr(self, "mirostat", None)
        if sampler is not None:
            # prohibit resetting sampler state if user has requested a continuation of the previous response
            sampler.can_reset = not no_msg
        prompt_tokens = dialog.endcode_turns(self.tokenizer, continuation=no_msg)
        input_total_len = len(prompt_tokens)  # needs only for `debug` statistics
        cache, idx = None, None
        if use_cache:  # means "full" cache here
            prompt_tokens = self.tokenizer.encode(dialog[-1].content, True, False)
            cache = getattr(self, "old_cache", None)
            pos = getattr(self, "old_pos", None)
            # pos is the same for all layers → get idx from first one
            idx = list(pos.values())[0] if pos is not None else None
            # cache exists and user wants continuation of Llama's answer
            if idx is not None and no_msg:
                # Single last token from Llama's previous reply
                prompt_tokens = [self.tokenizer.encode(dialog[-2].content, False, False)[-1]]
        total_len = len(prompt_tokens) + max_gen_len
        limit_len = total_len + 0 if idx is None else idx
        if limit_len > self.model.params.max_seq_len:
            print(f"The expected maximum content length {limit_len} exceeded the limit {self.model.params.max_seq_len} "
                  "(it is set by the `max_seq_len` parameter in `Llama.build()` and you can customise it to your liking).\n"
                  "All history was deleted and Dialog was restarted.")
            msg = dialog[-1].get_text()
            sys_msg = dialog.system_msg
            while len(dialog):
                dialog.pop()
            dialog.ensure_system(msg, sys_msg)
            dialog.just_initialized = True
            prompt_tokens = dialog.endcode_turns(self.tokenizer)
            total_len = len(prompt_tokens) + max_gen_len
        if dialog.just_initialized:
            cache, idx = None, None
            self.old_cache = None
            self.old_pos = None
            dialog.just_initialized = False
            torch.cuda.empty_cache()
        hooks, cache, pos = self.install_kv_cache_hooks(size=total_len, cache=cache, idx=idx)
        if use_cache:
            self.old_cache, self.old_pos = cache, pos
        return prompt_tokens, input_total_len, hooks

    def chat_completion(
        self,
        dialog: Dialog,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        mir_t: float = 0.0,
        mir_l: int = 0,
        logprobs: bool = False,
        show_bar: bool = False,
        use_cache: bool = False,
    ):
        if mir_t:
            sampler = getattr(self, "mirostat", None)
            if sampler is None:  # Create Sampler, when its first required
                sampler = MirostatSampler(tau=mir_t, eta=0.1)
                self.mirostat = sampler
            if sampler.tau != mir_t:  # New tau sets during chatting → recreate Sampler
                sampler = MirostatSampler(tau=mir_t, eta=0.1)
                self.mirostat = sampler
        prompt_tokens, dialog_len, hooks = self.manage_history(dialog, max_gen_len, use_cache)
        gen_tokens, gen_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            mir_t=mir_t,
            mir_l=mir_l,
            logprobs=logprobs,
            show_bar=show_bar,
        )
        for hook in hooks:
            hook.remove()
        torch.cuda.empty_cache()  # cache clean up
        if mir_t > 0:
            self.mirostat.reset()
        if len(gen_tokens) == 0:
            return None  # Happens whit eos at first generation step
        if logprobs:
            return {
                "generation": Message("assistant", self.tokenizer.decode(gen_tokens)),
                "tokens": gen_tokens,
                "logprobs": gen_logprobs,
                # Not sure if this is correct perplexity calculation
                "perplexity": f'{(-gen_logprobs).exp().mean():.2f}',
                "full_tokens": dialog_len + len(gen_tokens),
            }
        else:
            return {
                "generation": Message("assistant", self.tokenizer.decode(gen_tokens)),
                "tokens": gen_tokens,
                "full_tokens": dialog_len + len(gen_tokens),
            }


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class MirostatSampler(object):
    # Note: the implementation is based on `llama.cpp` and the official version from the `Mirostat` authors
    def __init__(self, tau: float = 5., eta: float = 1e-1):
        self.tau = tau
        self.eta = eta
        self.mu = 2 * tau
        self.can_reset = False
        # print(f"Initialized: mu {self.mu}")

    def get_k(self, probs: torch.Tensor, n_toks: int):
        n_vocab = probs.size(0)
        # Estimate s_hat using the most probable n_toks tokens
        idxs = torch.arange(0, n_toks, device=probs.device)
        nums = ((idxs + 2) / (idxs + 1)).log()
        prob = (probs[:n_toks] / probs[1:n_toks + 1]).log()
        sum_ti_bi = (nums * prob).sum()
        sum_ti_sq = (nums * nums).sum()
        s_hat = sum_ti_bi / sum_ti_sq
        # Compute k from the estimated s_hat and target surprise value
        eps = s_hat - 1
        # print(f"s_hat: {s_hat}, mu: {self.mu}, vocab: {n_vocab}")
        k = ((eps * 2**self.mu) / (1 - n_vocab**-eps)) ** (1 / s_hat)
        return max(1, int(k))

    def __call__(self, logits: torch.Tensor, n_toks: Optional[int] = None):
        assert logits.ndim == 1, "Single token only."
        logits, idxs = torch.sort(logits, descending=True)
        probs = logits.softmax(dim=0)
        if n_toks is not None:
            lim = self.get_k(probs, n_toks)
        else:
            # Truncate the tokens with surprise values greater than mu
            lim = max(1, (-probs.log2() < self.mu).sum())
        candidate = torch.multinomial(probs[:lim].softmax(0), 1)
        surprise = -probs[candidate].log2() - self.tau
        self.mu -= self.eta * surprise
        # print(f"lim={lim}, surprise-{surprise}, mu={self.mu}")
        return idxs[candidate]

    def reset(self):
        if self.can_reset:
            self.mu = self.tau * 2
