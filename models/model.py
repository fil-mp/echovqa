import os
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
import open_clip

from .llama_prompts import ModelArgs, Transformer
from .tokenizer import Tokenizer


class LLamaAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()

        # LLaMA model directory (must contain tokenizer.model + HF weights)
        hf_model_path = args.llm_model_path
        tokenizer_model_path = os.path.join(hf_model_path, "tokenizer.model")
        self.tokenizer = Tokenizer(model_path=tokenizer_model_path)

        self.model_dim = 4096  # LLaMA hidden dim

        model_args = ModelArgs(
            dim=self.model_dim,
            n_layers=args.llm_layers,
            n_heads=32,
            vocab_size=self.tokenizer.n_words,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.batch_size,
            w_lora=True,
            lora_rank=16,
        )

        from transformers import LlamaForCausalLM
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        try:
            print("Loading HuggingFace LLaMA model...")
            hf_model = LlamaForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            hf_state_dict = hf_model.state_dict()
            converted_weights = self.convert_hf_weights_to_adapter(hf_state_dict)
            self.llama.load_state_dict(converted_weights, strict=False)
            del hf_model
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            raise Exception("Failed to load LLaMA model")

        self.vision_model_type = args.vision_model
        self.test = getattr(args, "test", False)

        # Visual query processing
        self.query_len = args.query_len
        self.v_embed_dim = 768
        self.v_num_heads = 16
        self.v_mlp_ratio = 4.0
        self.v_depth = 8
        self.visual_query = nn.Embedding(args.query_len, self.v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(self.v_embed_dim, self.v_num_heads, self.v_mlp_ratio, qkv_bias=True)
            for _ in range(self.v_depth)
        ])
        self.visual_proj = nn.Linear(self.v_embed_dim, self.model_dim)
        self.visual_proj_norm = nn.LayerNorm(self.model_dim)

        if args.vision_model == "biomedclip":
            print("Using BiomedCLIP as the vision model...")
            self.biomedclip_model, self.biomedclip_preprocess = open_clip.create_model_from_pretrained(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            vision_hidden_size = 512
            self.clip_proj = nn.Linear(vision_hidden_size, self.v_embed_dim)
            self.clip_proj_norm = nn.LayerNorm(self.v_embed_dim)
        else:  # CLIP
            import clip
            self.clip, self.clip_transform = clip.load("ViT-L/14")
            vision_hidden_size = self.clip.visual.proj.shape[1]
            self.clip_proj = nn.Linear(vision_hidden_size, self.v_embed_dim).float()
            self.clip_proj_norm = nn.LayerNorm(self.v_embed_dim).float()

        # Kept for backward compatibility; primary path is clip_proj -> visual_blocks -> visual_proj
        self.feature_proj = nn.Linear(vision_hidden_size, self.model_dim, dtype=torch.bfloat16)
        self.feature_norm = nn.LayerNorm(self.model_dim)

        # Adapter settings
        self.adapter_percentage = args.adapter_percentage
        self.adapter_strategy = args.adapter_strategy
        total_layers = args.llm_layers
        self.num_adapter_layers = int(total_layers * self.adapter_percentage)
        self.adapter_query = nn.Embedding(self.num_adapter_layers * args.query_len, self.model_dim)

        # Deep prompts
        self.use_deep_prompts = args.use_deep_prompts
        if self.use_deep_prompts:
            self.num_deep_prompt_layers = args.num_deep_prompt_layers
            self.num_prompts = args.num_prompts
            self.prompt_dim = args.prompt_dim
            self.base_prompts = nn.Parameter(torch.zeros(self.num_prompts, self.prompt_dim))
            nn.init.normal_(self.base_prompts, std=0.02)
            self.prompt_projections = nn.ModuleList([
                nn.Sequential(nn.Linear(self.prompt_dim, self.model_dim), nn.Tanh())
                for _ in range(self.num_deep_prompt_layers)
            ])

        # Task head
        self.is_vqa = args.task_type == "vqa"
        if not self.is_vqa:
            self.num_classes = args.num_classes
            self.output_projection = nn.Linear(self.model_dim, self.num_classes)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.phase = args.phase
        self.get_trainable_params(self.phase)

    def get_projected_prompts(self, layer_idx):
        if not self.use_deep_prompts:
            return None
        return self.prompt_projections[layer_idx](self.base_prompts)

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if name.startswith("llama.layers.31") or 'norm' in name or 'bias' in name or 'lora' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

            trainable_components = [
                'feature_proj', 'feature_norm', 'adapter_query',
                'clip_proj', 'clip_proj_norm', 'visual_query',
                'visual_proj', 'visual_proj_norm',
            ]
            for name, para in self.named_parameters():
                if any(comp in name for comp in trainable_components):
                    para.data = para.data.float()
                    para.requires_grad = True

        elif phase == 'pretrain':
            total_layers = len(self.llama.layers)
            adapter_layer_indices = (
                range(total_layers - self.num_adapter_layers, total_layers)
                if self.adapter_strategy == 'late'
                else range(0, self.num_adapter_layers)
            )
            for i in adapter_layer_indices:
                layer_prefix = f"llama.layers.{i}"
                for name, para in self.named_parameters():
                    if name.startswith(layer_prefix) and 'gate' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

            trainable_components = [
                'feature_proj', 'feature_norm', 'adapter_query',
                'clip_proj', 'clip_proj_norm', 'visual_query',
                'visual_proj', 'visual_proj_norm',
                'visual_blocks',
                'base_prompts', 'prompt_projections',
            ]
            for name, para in self.named_parameters():
                if any(comp in name for comp in trainable_components):
                    para.data = para.data.float()
                    para.requires_grad = True

            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
                        print(f"Training LLaMA parameter: {name}, Shape: {para.shape}")
        else:
            raise ValueError(f"Unknown model phase: {phase}")

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def convert_hf_weights_to_adapter(self, hf_state_dict):
        new_state_dict = {}
        key_mappings = {
            'model.embed_tokens.weight': 'tok_embeddings.weight',
            'model.norm.weight': 'norm.weight',
            'lm_head.weight': 'output.weight',
        }
        layer_mappings = {
            'input_layernorm.weight': 'attention_norm.weight',
            'post_attention_layernorm.weight': 'ffn_norm.weight',
            'self_attn.q_proj.weight': 'attention.wq.weight',
            'self_attn.k_proj.weight': 'attention.wk.weight',
            'self_attn.v_proj.weight': 'attention.wv.weight',
            'self_attn.o_proj.weight': 'attention.wo.weight',
            'mlp.gate_proj.weight': 'feed_forward.w1.weight',
            'mlp.down_proj.weight': 'feed_forward.w2.weight',
            'mlp.up_proj.weight': 'feed_forward.w3.weight',
        }
        for key, value in hf_state_dict.items():
            if 'model.layers.' in key:
                parts = key.split('.')
                layer_num = parts[2]
                sub_key = '.'.join(parts[3:])
                if sub_key.endswith('.weight'):
                    for old, new in layer_mappings.items():
                        if old in sub_key:
                            new_state_dict[f'layers.{layer_num}.{new}'] = value
                            break
            else:
                for old, new in key_mappings.items():
                    if old in key:
                        new_state_dict[new] = value
                        break
        return new_state_dict

    def forward_visual(self, pixel_values):
        if self.vision_model_type in ("biomedclip", "biomedclip_simpool"):
            with torch.cuda.amp.autocast(dtype=torch.float32):
                clip_feats = self.biomedclip_model.encode_image(pixel_values)
            proj_dtype = self.clip_proj.weight.dtype
            clip_feats = clip_feats.to(proj_dtype)
            clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats))

            visual_query = self.visual_query.weight.unsqueeze(0).repeat(len(pixel_values), 1, 1)
            visual_query = torch.cat([visual_query, clip_feats.unsqueeze(1)], dim=1)
            for block in self.visual_blocks:
                visual_query = block(visual_query)
            visual_query = visual_query[:, :self.query_len, :]
            visual_query = self.visual_proj(visual_query)
            visual_query = self.visual_proj_norm(visual_query)
            return visual_query
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                clip_feats = self.clip_encode_image(pixel_values)
            proj_dtype = self.clip_proj.weight.dtype
            clip_feats = self.clip_proj(clip_feats.to(proj_dtype))
            clip_feats = self.clip_proj_norm(clip_feats)

            visual_query = self.visual_query.weight.unsqueeze(0).repeat(len(pixel_values), 1, 1)
            visual_query = torch.cat([visual_query, clip_feats], dim=1)
            for block in self.visual_blocks:
                visual_query = block(visual_query)
            visual_query = visual_query[:, :self.query_len, :]
            visual_query = self.visual_proj(visual_query)
            visual_query = self.visual_proj_norm(visual_query)
            return visual_query

    def clip_encode_image(self, x):
        x = x.to(self.clip.visual.conv1.weight.dtype)
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = x.to(torch.float32)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.visual.ln_post(x[:, :, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        visual_query = self.forward_visual(pixel_values)
        tokens = input_ids
        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        total_layers = len(self.llama.layers)
        adapter_layer_indices = set(
            range(total_layers - self.num_adapter_layers, total_layers)
            if self.adapter_strategy == 'late'
            else range(self.num_adapter_layers)
        )
        adapter = self.adapter_query.weight.reshape(
            self.num_adapter_layers, self.query_len, -1).unsqueeze(1)

        for i, layer in enumerate(self.llama.layers):
            if i in adapter_layer_indices:
                adapter_idx = i if self.adapter_strategy == 'early' else i - (total_layers - self.num_adapter_layers)
                dynamic_adapter = adapter[adapter_idx].repeat(_bsz, 1, 1) + visual_query
                if self.use_deep_prompts and i < self.num_deep_prompt_layers:
                    projected_prompts = self.get_projected_prompts(i)
                    k = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    v = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    extended_mask = torch.full((1, 1, h.size(1), k.size(1)), float("-inf"), device=h.device)
                    extended_mask = torch.triu(extended_mask, diagonal=1).type_as(h)
                    h = layer(h, 0, freqs_cis, extended_mask, dynamic_adapter, k=k, v=v)
                else:
                    h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            else:
                if self.use_deep_prompts and i < self.num_deep_prompt_layers:
                    projected_prompts = self.get_projected_prompts(i)
                    k = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    v = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    extended_mask = torch.full((1, 1, h.size(1), k.size(1)), float("-inf"), device=h.device)
                    extended_mask = torch.triu(extended_mask, diagonal=1).type_as(h)
                    h = layer(h, 0, freqs_cis, extended_mask, k=k, v=v)
                else:
                    h = layer(h, 0, freqs_cis, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
        return c_loss

    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos: start_pos + seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        total_layers = len(self.llama.layers)
        adapter_layer_indices = set(
            range(total_layers - self.num_adapter_layers, total_layers)
            if self.adapter_strategy == 'late'
            else range(self.num_adapter_layers)
        )
        adapter = self.adapter_query.weight.reshape(
            self.num_adapter_layers, self.query_len, -1).unsqueeze(1)

        for i, layer in enumerate(self.llama.layers):
            if i in adapter_layer_indices:
                adapter_idx = i if self.adapter_strategy == 'early' else i - (total_layers - self.num_adapter_layers)
                dynamic_adapter = adapter[adapter_idx].repeat(_bsz, 1, 1) + visual_query
                if self.use_deep_prompts and i < self.num_deep_prompt_layers:
                    projected_prompts = self.get_projected_prompts(i)
                    k = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    v = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    extended_mask = torch.full((1, 1, h.size(1), k.size(1)), float("-inf"), device=h.device)
                    extended_mask = torch.triu(extended_mask, diagonal=start_pos + 1).type_as(h)
                    h = layer(h, start_pos, freqs_cis, extended_mask, dynamic_adapter, k=k, v=v)
                else:
                    h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            else:
                if self.use_deep_prompts and i < self.num_deep_prompt_layers:
                    projected_prompts = self.get_projected_prompts(i)
                    k = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    v = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    extended_mask = torch.full((1, 1, h.size(1), k.size(1)), float("-inf"), device=h.device)
                    extended_mask = torch.triu(extended_mask, diagonal=start_pos + 1).type_as(h)
                    h = layer(h, start_pos, freqs_cis, extended_mask, k=k, v=v)
                else:
                    h = layer(h, start_pos, freqs_cis, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])
        return output.float()

    @torch.inference_mode()
    def generate(self, pixel_values, prompts, max_gen_len: int = 256,
                 temperature: float = 0.1, top_p: float = 0.75):
        bsz = len(pixel_values)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(pixel_values) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(pixel_values)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        max_prompt_size = max([len(t) for t in prompts])
        min_prompt_size = min([len(t) for t in prompts])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def generate_with_pseudo_tokens(self, pixel_values, prompts, pseudo_tokens, num_tokens=1, **kwargs):
        """Generate using optimized embeddings injected at '$' token positions."""
        visual_query = self.forward_visual(pixel_values)
        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        prompts_tensor = torch.tensor(prompts).cuda()
        input_embeds = self.llama.tok_embeddings(prompts_tensor)

        dollar_token_id = self.tokenizer.encode("$", bos=False, eos=False)[0]
        for b in range(prompts_tensor.shape[0]):
            dollar_positions = (prompts_tensor[b] == dollar_token_id).nonzero(as_tuple=True)[0]
            if len(dollar_positions) != pseudo_tokens.shape[0]:
                print(f"Warning: Sample {b} has {len(dollar_positions)} `$` tokens, "
                      f"but {pseudo_tokens.shape[0]} pseudo tokens provided")
            for i, pos in enumerate(dollar_positions):
                if i < pseudo_tokens.shape[0]:
                    input_embeds[b, pos] = pseudo_tokens[i].to(input_embeds.dtype)

        return self._generate_with_embeddings(visual_query, input_embeds, **kwargs)

    @torch.inference_mode()
    def _generate_with_embeddings(self, visual_query, input_embeds,
                                  max_gen_len=50, temperature=0.1, top_p=0.75, max_retries=3):
        """Generate from precomputed input embeddings, retrying if EOS is the first token."""
        _bsz, seqlen, _ = input_embeds.shape
        params = self.llama.params
        total_len = min(params.max_seq_len, max_gen_len + seqlen)

        eos_generated_first = False
        cur_pos = seqlen
        tokens = None
        for attempt in range(max_retries):
            all_embeds = torch.zeros((_bsz, total_len, input_embeds.shape[-1]),
                                     dtype=input_embeds.dtype, device=input_embeds.device)
            all_embeds[:, :seqlen] = input_embeds
            tokens = torch.full((_bsz, total_len), self.tokenizer.pad_id,
                                dtype=torch.long, device=input_embeds.device)

            start_pos = seqlen
            prev_pos = 0
            eos_generated_first = False
            for cur_pos in range(start_pos, total_len):
                current_embeds = all_embeds[:, prev_pos:cur_pos]
                with torch.cuda.amp.autocast():
                    logits = self._forward_inference_with_embeds(visual_query, current_embeds, prev_pos)
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = self.sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits, dim=-1)
                next_token = next_token.reshape(-1)
                next_embed = self.llama.tok_embeddings(next_token.unsqueeze(0))
                all_embeds[:, cur_pos] = next_embed.squeeze(0)
                tokens[:, cur_pos] = next_token

                if cur_pos == seqlen and next_token[0].item() == self.tokenizer.eos_id:
                    eos_generated_first = True
                    break
                if _bsz == 1 and next_token[0].item() == self.tokenizer.eos_id:
                    break
                prev_pos = cur_pos

            if not eos_generated_first:
                break

        if eos_generated_first:
            return [""], True

        decoded = []
        for i in range(_bsz):
            gen_tokens = tokens[i, seqlen:cur_pos + 1].tolist()
            try:
                eos_idx = gen_tokens.index(self.tokenizer.eos_id)
                gen_tokens = gen_tokens[:eos_idx]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(gen_tokens))
        return decoded, False

    @torch.inference_mode()
    def _forward_inference_with_embeds(self, visual_query, input_embeds, start_pos: int):
        _bsz, seqlen, _ = input_embeds.shape
        h = input_embeds
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos: start_pos + seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        total_layers = len(self.llama.layers)
        adapter_layer_indices = set(
            range(total_layers - self.num_adapter_layers, total_layers)
            if self.adapter_strategy == 'late'
            else range(self.num_adapter_layers)
        )
        adapter = self.adapter_query.weight.reshape(
            self.num_adapter_layers, self.query_len, -1).unsqueeze(1)

        for i, layer in enumerate(self.llama.layers):
            if i in adapter_layer_indices:
                adapter_idx = i if self.adapter_strategy == 'early' else i - (total_layers - self.num_adapter_layers)
                dynamic_adapter = adapter[adapter_idx].repeat(_bsz, 1, 1) + visual_query
                if self.use_deep_prompts and i < self.num_deep_prompt_layers:
                    projected_prompts = self.get_projected_prompts(i)
                    k = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    v = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    extended_mask = torch.full((1, 1, h.size(1), k.size(1)), float("-inf"), device=h.device)
                    extended_mask = torch.triu(extended_mask, diagonal=start_pos + 1).type_as(h)
                    h = layer(h, start_pos, freqs_cis, extended_mask, dynamic_adapter, k=k, v=v)
                else:
                    h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            else:
                if self.use_deep_prompts and i < self.num_deep_prompt_layers:
                    projected_prompts = self.get_projected_prompts(i)
                    k = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    v = torch.cat([projected_prompts.unsqueeze(0).repeat(_bsz, 1, 1), h], dim=1)
                    extended_mask = torch.full((1, 1, h.size(1), k.size(1)), float("-inf"), device=h.device)
                    extended_mask = torch.triu(extended_mask, diagonal=start_pos + 1).type_as(h)
                    h = layer(h, start_pos, freqs_cis, extended_mask, k=k, v=v)
                else:
                    h = layer(h, start_pos, freqs_cis, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])
        return output.float()
