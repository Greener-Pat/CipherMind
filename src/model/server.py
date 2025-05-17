import pickle
import socket
from ciphermind import CipherMindModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel

def receive_large_data(sock, chunk_size=1024):
    """
    从socket接收大块数据的分片处理

    Args:
        sock (socket.socket): 已连接的socket对象
        chunk_size (int, optional): 分片接收大小，默认1024字节

    Returns:
        bytes: 完整接收的二进制数据

    Raises:
        ConnectionError: 当socket连接异常中断时抛出
    """
    received_data = b""  # 存储接收到的数据
    while True:
        chunk = sock.recv(chunk_size)  # 接收分片数据
        received_data += chunk
        if len(chunk) != chunk_size:
            break
    return received_data

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tunned_model_name = "../../data/models/tunning_15_0"
model = AutoModelForCausalLM.from_pretrained(tunned_model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 6666))  # 0.0.0.0 表示监听所有网络接口
server_socket.listen(5)  # 开始监听，最多允许 5 个连接排队

server_model = CipherMindModel(model, tokenizer)

print("等待客户端连接...")
client_socket, client_addr = server_socket.accept()  # 阻塞等待客户端连接
print(f"已连接客户端：{client_addr}")

s = ""
while True:
    data = receive_large_data(client_socket)  # 接收数据，缓冲区大小为 1024 字节
    if not data:
        break

    data_tuple = pickle.loads(data)
    hidden_states = data_tuple[0]
    state = data_tuple[1]
    
    if state < 0:
        if state == -1:
            print(s)
        server_model.receiver_reset()
        continue
    
    s = server_model.receiver_step(hidden_states)
    client_socket.send("OK".encode('utf-8'))

client_socket.close()
server_socket.close()