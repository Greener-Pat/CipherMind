# CipherMind

An encrypted communication method based on LLM

<<<<<<< Updated upstream

## Function

- model

  - ciphermind.py - core model
  - server.py & client.py - receiver and sender, run in command line to transmit text
  - *test.py* - to test a model's repeat ability (tempporary exists)
  - collision.py - to test the collision rate (describe in cosine similarity pattern) of the base model, output its result to ./data/res/collision
- experiment

  - mmlu.py - apply the mmlu test on the given base model and tunned(lora) model, output its result to ./data/res/mmlu
  - correctness.py - to test the tranmission ability of the base model and tunned(lora) model, output its result to ./data/res/correctness
  - show.py - visualize the result of three test, namely the collision, correctness, mmlu

## Usage

> Enter corresponding directory first

- model

  - server & client

    ```sh
    # one terminal
    python servere.py
    # another terminal
    python client.py
    ```
  - collision

    ```sh
    python collision.py
    ```
- experiment

  - mmlu

    ```sh
    python mmlu.py
    ```
  - correctness

    ```sh
    python correctness.py
    ```
  - visualize

    ```sh
    python show.py
    ```

=======
文件组织:

```
CipherMind
|- data
    |- mmlu
	|- test
	    |- data-00000-of-00001.arrow
	    |- dataset_info.json
	    |- state.json
    |- res
	|- collision
	    |- collision_char.pkl
	    |- collision.pkl
	|- correctness
	    |- base_map.pkl
	    |- lora_map.pkl
	|- mmlu
	    |- base_mmlu.pkl
	    |- lora_mmlu.pkl
    |- text
	|- news_data.txt
	|- small_news.txt
|- src
    |- experiment
	|- correctness.py
	|- mmlu.py
	|- read_mmlu.py
	|- show.py
    |- model
	|- __init__.py
	|- ciphermind.py
	|- client.py
	|- collision.py
	|- server.py
	|- test.py
    |- tunning
	|- tunning.py
    |- __init__.py
|- README.md
```

## src

具体的实现代码

### experiment

实验用代码

### model

模型主要架构代码

1. ciphermind.py
   其中有主要的类CipherMind,每次进行通讯时，两方都需要生成一个CipherMind类实例，且生成时需要选择相同的model和tokenizer。
   其中有四类实现:推理、中间层提取、随机数种子设置、单步接受&发送函数。
2. client.py
   客户端类，用于生成客户端实例，其中有两类函数:

   1. 生成随机数种子函数，用于生成客户端的随机数种子。
   2. 单步接受&发送函数，用于客户端接受&发送消息。
3. server.py
   服务端类，用于生成服务端实例，其中有两类函数:

   1. 生成随机数种子函数，用于生成服务端的随机数种子。
   2. 单步接受&发送函数，用于服务端接受&发送消息。

### tunning

确定性微调代码

> Stashed changes

- 确定性转化为可信性
- 大模型水印技术，水印增强技术，可窃听、可破译、不可混入
- 可破解性达到硬件级——自我训练类脑芯片：一直在芯片上进行读写，不和内存进行交换、局部反向传播、分布式反向传播；封装后成为黑盒
