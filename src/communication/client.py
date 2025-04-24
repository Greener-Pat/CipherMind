import pickle
import socket
from model.ciphermind import CipherMindModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def send_large_data(sock, data, chunk_size=1024):
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:total_sent + chunk_size])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

client_model = CipherMindModel(model, tokenizer)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.3.108', 6666))  # 连接服务端

# 发送数据
while True:
    to_send = input("user>")
    if to_send == "q" or to_send == "quit":
        break

    messages=[{"role": "system", "content": "你是一个复读机"}, {"role": "user", "content": "'" + to_send + "', 请重复一遍"}]

    input_ids = client_model.init_input_ids(messages)
    idx = 0
    while True:
        hidden_states, out_layer, input_ids = client_model.sender_step(input_ids, idx)
        idx += 1
        if input_ids is None:
            # send end signal
            data_tuple = pickle.dumps((hidden_states, -1))
            send_large_data(client_socket, data_tuple)
            break
        
        print(f"layer: {out_layer}")
        data_tuple = pickle.dumps((hidden_states, out_layer))
        send_large_data(client_socket, data_tuple)

        # 接收响应
        response = client_socket.recv(1024)
client_socket.close()