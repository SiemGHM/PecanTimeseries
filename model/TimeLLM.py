from math import sqrt
import os

import torch
import torch.nn as nn

from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer, 
    GPT2Config, GPT2Model, GPT2Tokenizer, 
    BertConfig, BertModel, BertTokenizer
)
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.cuda.amp

from layers.EmbedPatch import PatchEmbedding
from layers.StandardNorm import Normalize
from huggingface_hub import login
import transformers

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
# Suppress transformers logging
transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        print(configs.d_ff, "######################################################")
        configs.d_ff = 768
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        print("CONFIG>LLMMOD", configs.llm_model)

        # --- Load the LLM ---
        if configs.llm_model in ['LLAMA', 'LLAMAOrig']:
            self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                print("Here here")
                self.llm_model = LlamaModel.from_pretrained(
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    torch_dtype=torch.bfloat16,
                     # device_map="auto",
                    # load_in_4bit=True
                )
                print("loaded")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    torch_dtype=torch.bfloat16,
                   # device_map="auto",
                    # load_in_4bit=True
                )
                print("loaded sec attempt")
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'meta-llama/Llama-2-7b-hf',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )

                # self.llm_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")



            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        # --- Set tokenizer pad token ---
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = ('The Electricity Transformer Temperature (ETT) is a crucial indicator '
                                'in the electric power long-term deployment.')

        self.dropout = nn.Dropout(configs.dropout)

        # --- Create custom layers (device and dtype will be set by Accelerate) ---
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)
                                    
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Assume inputs are already on the correct device (Accelerate will handle this)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # Calculate input statistics
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # Build prompt for each example
        prompt_list = []
        for b in range(x_enc.shape[0]):
            min_str = str(min_values[b].item())
            max_str = str(max_values[b].item())
            median_str = str(medians[b].item())
            lags_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description} "
                f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps; "
                f"Input statistics: min value {min_str}, max value {max_str}, median value {median_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_str}<|<end_prompt>|>"
            )
            prompt_list.append(prompt_)

        # Reshape x_enc back to (B, T, N)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # Tokenize prompt (do not explicitly move to device)
        prompt = self.tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.llm_model.get_input_embeddings().weight.device))

        # Compute source embeddings via mapping_layer
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # Process through patch embedding and reprogramming layers
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc = x_enc.to(torch.bfloat16)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Concatenate prompt embeddings with patch-embedded output

        down_project = nn.Linear(4096, 768).to(enc_out.device).to(torch.bfloat16)

        # Down-project enc_out.
        enc_out_projected = down_project(enc_out)  # Shape: [B, L_enc, 768]

        # Now both prompt_embeddings and enc_out_projected have the same feature dimension.
        llama_enc_out = torch.cat([prompt_embeddings, enc_out_projected], dim=1)
        # Shape of llama_enc_out becomes: [B, L_prompt + L_enc, 768]

        # Feed the combined embeddings to GPT‑2.
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state

        # (Optional) Trim or process dec_out as needed.
        dec_out = dec_out[:, :, :self.d_ff]

        # Reshape and project the output
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # print("dec_out shape before slicing:", dec_out.shape)
        sliced_dec_out = dec_out[:, :, :, -self.patch_nums:]
        # print("Sliced dec_out shape:", sliced_dec_out.shape)
        # flattened = sliced_dec_out.view(sliced_dec_out.size(0), -1)
        # print("Flattened dec_out shape:", flattened.shape)

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out

    def calcute_lags(self, x_enc):
        # Explicitly cast x_enc to float32.
        x_enc_float = x_enc.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            q_fft = torch.fft.rfft(x_enc_float.permute(0, 2, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(x_enc_float.permute(0, 2, 1).contiguous(), dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)
            mean_value = torch.mean(corr, dim=1)
            _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def calcute_lagsOrig(self, x_enc):
        x_enc_float = x_enc.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)
            mean_value = torch.mean(corr, dim=1)
            _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        d_llm_2 = 768
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm_2, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm_2, d_keys * n_heads) #d_llm
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    

    def forwardOrig(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Project the inputs and print their shapes.
        target_proj = self.query_projection(target_embedding)
        # print("target_proj shape:", target_proj.shape)  # Expected: (B, L, d_keys * n_heads)
        
        source_proj = self.key_projection(source_embedding)
        # print("source_proj shape:", source_proj.shape)  # Expected: (S, d_keys * n_heads)
        
        value_proj = self.value_projection(value_embedding)
        # print("value_proj shape:", value_proj.shape)  # Expected: (S, d_keys * n_heads)

        # Reshape to add the heads dimension.
        target_embedding = target_proj.view(B, L, H, -1)
        source_embedding = source_proj.view(S, H, -1)
        value_embedding = value_proj.view(S, H, -1)

        # print("Reshaped target_embedding:", target_embedding.shape)  # (B, L, H, ?)
        # print("Reshaped source_embedding:", source_embedding.shape)  # (S, H, ?)
        # print("Reshaped value_embedding:", value_embedding.shape)  # (S, H, ?)

        # Continue with the reprogramming operation.
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        # print("Output of reprogramming shape:", out.shape)  # Expected: (B, L, H, ?)

        out = out.reshape(B, L, -1)
        # print("Output after reshape:", out.shape)  # Expected: (B, L, d_keys * n_heads)
        
        final_out = self.out_projection(out)
        # print("Final output shape:", final_out.shape)  # Expected: (B, L, d_llm)
        
        return final_out
