import pickle
import socket
from part2.ciphermind import CipherMindModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def receive_large_data(sock, chunk_size=1024):
    received_data = b""  # 存储接收到的数据
    while True:
        chunk = sock.recv(chunk_size)  # 接收分片数据
        received_data += chunk
        if len(chunk) != chunk_size:
            break
    return received_data

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 6666))  # 0.0.0.0 表示监听所有网络接口
server_socket.listen(5)  # 开始监听，最多允许 5 个连接排队

server_model = CipherMindModel(model, tokenizer)

print("等待客户端连接...")
client_socket, client_addr = server_socket.accept()  # 阻塞等待客户端连接
print(f"已连接客户端：{client_addr}")

start = True
s = ""
while True:
    data = receive_large_data(client_socket)  # 接收数据，缓冲区大小为 1024 字节
    if not data:
        break

    data_tuple = pickle.loads(data)
    hidden_states = data_tuple[0]
    out_layer = data_tuple[1]
    
    if out_layer == -1:
        print(s)
        server_model.receiver_reset()
        continue
    
    s = server_model.receiver_step(hidden_states, out_layer)
    start = False
    client_socket.send("OK".encode('utf-8'))

client_socket.close()
server_socket.close()