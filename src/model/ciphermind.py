import torch
import random
import torch.nn.functional as F
import string

def random_string(length=10):
    chars = string.ascii_letters + string.digits  # 字母+数字
    return ''.join(random.choices(chars, k=length))

class CipherMindModel():
    def __init__(self, model, tokenizer):
        """初始化模型组件并配置计算设备
        Args:
            model: 预训练语言模型实例
            tokenizer: 分词器实例
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(self.device)
        model.eval()
        self.tokenizer = tokenizer
        self.config = model.model.config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers.to(self.device)
        self.norm = model.model.norm.to(self.device)
        self.lm_head = model.lm_head.to(self.device)
        self.inv_freq = model.model.rotary_emb.inv_freq.to(self.device)
        self.layer_num = len(model.model.layers)

        self.max_length = 128       # 以此作padding
        self.generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        self.finish = False
        self.to_send = None         # 预期发送token
        self.middle_layer = 3    # 预期中间层(初始层数由最初交换的密钥决定)
        self.seed = 0               # 制造随机层数分布的种子
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        assert 0 <= self.middle_layer < len(self.layers)
    
    def update_random(self,raw_state):
        """更新随机种子，确保每次生成的随机数序列不同"""
        random_index = torch.randint(0, raw_state.shape[1], (1,)).item()  # 生成0-26的随机整数
        # print(f"随机选择第{random_index}个切片，形状：{raw_state.shape}")
        assert 0 <= random_index < raw_state.shape[1]
        selected_slice = raw_state[:, random_index, :]  # 保持第一个和第三个维度，切片第二个维度
        # print(f"要增加的量:{int(selected_slice.abs().sum().item())}")
        self.seed += int(selected_slice.abs().sum().item())
        # print(f"当前种子:{self.seed}")
        torch.manual_seed(self.seed)# 令下一次选择的切片不同
        random.seed(self.seed)# 令下一次选择的层数不同
        self.middle_layer = random.randint(0, self.layer_num - 1)
        assert 0 <= self.middle_layer < len(self.layers)
    
    def rotary_emb(self, position_ids):
        """生成旋转位置编码(RoPE)的余弦/正弦分量
        Args:
            position_ids: 位置ID张量
        Returns:
            tuple: (cos_emb, sin_emb) 位置嵌入对
        """
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

    def encode(self, input_ids):
        """编码过程：从输入ID到指定层的隐藏状态
        Args:
            input_ids: 输入token ID序列
        Returns:
            torch.Tensor: 指定层的隐藏状态
        """
        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)

        # 位置编码
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[:self.middle_layer]:
            hidden_states = decoder_layer.forward(
                hidden_states,
                position_ids=position_ids,
                output_attentions=self.config.output_attentions,
                use_cache=self.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        return hidden_states
    
    def decode(self, hidden_states):
        """解码过程：从指定层开始生成后续隐藏状态直到最后一层
        Args:
            hidden_states: 初始隐藏状态
        Returns:
            torch.Tensor: 最终解码后的隐藏状态
        """
        cache_position = torch.arange(0, len(hidden_states[0]), device=self.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids)

        for decoder_layer in self.layers[self.middle_layer:]:
            hidden_states = decoder_layer.forward(
                hidden_states,
                position_ids=position_ids,
                output_attentions=self.config.output_attentions,
                use_cache=self.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def decode_for_experiment(self, out_layer, hidden_states):
        """解码过程：从指定层开始生成后续隐藏状态直到最后一层
        Args:
            out_layer: 输出层, 用于实验
            hidden_states: 初始隐藏状态
        Returns:
            torch.Tensor: 最终解码后的隐藏状态
        """
        cache_position = torch.arange(0, len(hidden_states[0]), device=self.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(position_ids)

        for decoder_layer in self.layers[out_layer:]:
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
        """生成用于触发模型重复行为的初始化输入序列
        Args:
            to_send: 需要模型重复的目标字符串
        Returns:
            torch.Tensor: 初始化后的输入ID序列
        """
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
        """将隐藏状态转换为预测token ID
        Args:
            hidden_states: 隐藏状态张量
            temperature: 采样温度参数
        Returns:
            torch.Tensor: 下一个预测token ID
        """
        # 应用语言模型头获取logits
        logits = self.lm_head(hidden_states[:, -1, :])  # 取最后一个位置的隐藏状态
        probs = F.softmax(logits / temperature, dim=-1)
        
        # 选择下一个token（示例使用贪心搜索）
        next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
        return next_token_id

    def sender_step(self, input_ids, idx):
        """发送端单步处理流程
        Args:
            input_ids: 当前输入序列
            idx: 当前处理步骤索引
        Returns:
            tuple: (中间状态, 传输状态, 更新后的输入序列)
            传输状态说明:
                0: 正常传输中
                -1: 成功&填充完毕&传输完毕
                -2: 得到了多余的token
                -3: 要通讯的内容超出最大长度
        """
        if idx == self.max_length:  # 要通讯的内容超出最大长度
            # reset
            self.finish = False
            if self.to_send_id < len(self.to_send[0]):
                # fail to send
                return None, -3, None
            else:
                # 传输成功并填充完毕
                return None, -1, None
        
        if self.finish:
            # random padding
            rand_s = random_string(10)
            input_ids = self.tokenizer(rand_s, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = input_ids.to(self.device)
        
        # 在本地计算出中间状态+最终结果
        middle_states = self.encode(input_ids)
        final_states = self.decode(middle_states)
        next_token_id = self.token_translate(final_states)
        
        if next_token_id == self.tokenizer.eos_token_id:
            self.finish = True
            if self.to_send_id < len(self.to_send[0]):
                # token缺失
                print("fail to send")

        if self.finish:
            self.update_random(final_states)
            # 不做判断直接返回随机生成的token
            return middle_states, 0, input_ids

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        if self.to_send_id >= len(self.to_send[0]) or next_token_id[0][0] != self.to_send[0][self.to_send_id]:
            # 认为预测得到多余的token
            return middle_states, -2, input_ids
        else:
            self.to_send_id += 1
        self.update_random(final_states)
        return middle_states, 0, input_ids

    def receiver_step(self, hidden_states):
        """接收端单步解码处理
        Args:
            hidden_states: 中间隐藏状态
        Returns:
            str: 解码生成的字符串
        """
        final_states = self.decode(hidden_states)
        self.update_random(final_states)
        next_token_id = self.token_translate(final_states)

        if next_token_id == self.tokenizer.eos_token_id:
            self.finish = True
        if not self.finish:
            # only the valid token is added to the generated_ids
            self.generated_ids = torch.cat([self.generated_ids, next_token_id], dim=-1)
        s = self.tokenizer.decode(self.generated_ids[0], skip_special_tokens=True)
        return s

    def receiver_step_for_experiment(self, hidden_states,out_layer):
        """接收端单步解码处理
        Args:
            hidden_states: 中间隐藏状态
            out_layer: 输出的中间层,用于碰撞测试
        Returns:
            str: 解码生成的字符串
        """
        final_states = self.decode_for_experiment(out_layer, hidden_states)
        next_token_id = self.token_translate(final_states)
        if next_token_id == self.tokenizer.eos_token_id:
            self.finish = True

        if not self.finish:
            # only the valid token is added to the generated_ids
            self.generated_ids = torch.cat([self.generated_ids, next_token_id], dim=-1)
        s = self.tokenizer.decode(self.generated_ids[0], skip_special_tokens=True)
        return s

    def receiver_reset(self):
        """重置接收端生成状态"""
        self.generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        self.finish = False