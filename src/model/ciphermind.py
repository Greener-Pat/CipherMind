import torch
import random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import string

def random_string(length=10):
    chars = string.ascii_letters + string.digits  # 字母+数字
    return ''.join(random.choices(chars, k=length))

class CipherMindModel():
    def __init__(self, model, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.config = model.model.config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers.to(self.device)
        self.norm = model.model.norm.to(self.device)
        self.lm_head = model.lm_head.to(self.device)
        self.inv_freq = model.model.rotary_emb.inv_freq.to(self.device)
        self.layer_num = len(model.model.layers)

        self.max_length = 16        # 以此作padding
        self.generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        self.finish = False
        self.to_send = None         # 预期发送token

    def rotary_emb(self, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = self.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)

    def encode(self, input_ids, out_layer):
        assert 0 <= out_layer < len(self.layers)
        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)

        # 位置编码
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[:out_layer]:
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                output_attentions=self.config.output_attentions,
                use_cache=self.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        return hidden_states
    
    def decode(self, in_layer, hidden_states):
        cache_position = torch.arange(0, len(hidden_states[0]), device=self.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids)

        for decoder_layer in self.layers[in_layer:]:
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                output_attentions=self.config.output_attentions,
                use_cache=self.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def init_input_ids(self, to_send):
        messages=[{"role": "system", "content": "You are a repeater"}, {"role": "user", "content": "Repeat in the same case, ' " + to_send + " '"}]
        # 添加相关符号
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 将文本转换为模型输入格式
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        self.to_send = self.tokenizer(to_send, return_tensors="pt").input_ids.to(self.device)
        self.to_send_id = 0
        return input_ids

    def token_translate(self, hidden_states, temperature=1.0):
        # 应用语言模型头获取logits
        logits = self.lm_head(hidden_states[:, -1, :])  # 取最后一个位置的隐藏状态
        probs = F.softmax(logits / temperature, dim=-1)
        
        # 选择下一个token（示例使用贪心搜索）
        next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
        return next_token_id

    def sender_step(self, input_ids, idx):
        if idx == self.max_length:
            # reset
            self.finish = False
            if self.to_send_id < len(self.to_send[0]):
                # fail to send
                return None, -3, None
            else:
                return None, -1, None
            
        
        if self.finish:
            rand_s = random_string(10)
            input_ids = self.tokenizer(rand_s, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = input_ids.to(self.device)
        out_layer = random.randint(0, self.layer_num - 1)
        middle_states = self.encode(input_ids, out_layer)
        final_states = self.decode(out_layer, middle_states)

        next_token_id = self.token_translate(final_states)
        
        if next_token_id == self.tokenizer.eos_token_id:
            self.finish = True
            if self.to_send_id < len(self.to_send[0]):
                print("fail to send")
        if self.finish:
            # 不做判断直接返回随机生成的token
            return middle_states, out_layer, input_ids

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        if self.to_send_id >= len(self.to_send[0]) or next_token_id[0][0] != self.to_send[0][self.to_send_id]:
            # 认为预测得到多余的token
            return None, -2, input_ids
        else:
            self.to_send_id += 1

        return middle_states, out_layer, input_ids

    def receiver_step(self, hidden_states, in_layer):
        hidden_states = self.decode(in_layer, hidden_states)
        next_token_id = self.token_translate(hidden_states)

        if next_token_id == self.tokenizer.eos_token_id:
            self.finish = True

        if not self.finish:
            # only the valid token is added to the generated_ids
            self.generated_ids = torch.cat([self.generated_ids, next_token_id], dim=-1)
        s = self.tokenizer.decode(self.generated_ids[0], skip_special_tokens=True)
        return s

    def receiver_reset(self):
        self.generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        self.finish = False

# D:\Computer_Download\anaconda\Lib\site-packages\transformers\models