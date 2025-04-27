import pickle
import socket
from ciphermind import CipherMindModel
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
print("Linking...")
client_socket.connect(('localhost', 6666))  # 连接服务端

# 发送数据
while True:
    to_send = input("user>")
    if to_send == "q" or to_send == "quit":
        break

    input_ids = client_model.init_input_ids(to_send)
    print("Sending...")
    idx = 0
    while True:
        hidden_states, out_layer, input_ids = client_model.sender_step(input_ids, idx)
        if out_layer == -2:
            continue
        idx += 1
        if out_layer < 0:
            # send end signal
            data_tuple = pickle.dumps((hidden_states, out_layer))
            send_large_data(client_socket, data_tuple)
            break
        
        data_tuple = pickle.dumps((hidden_states, out_layer))
        send_large_data(client_socket, data_tuple)

        # 接收响应
        response = client_socket.recv(1024)
client_socket.close()