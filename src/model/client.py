import pickle
import socket
from ciphermind import CipherMindModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel

def send_large_data(sock, data, chunk_size=1024):
    """
    发送大数据的分片处理

    Args:
        sock (socket.socket): 已连接的socket对象
        data (bytes): 需要发送的二进制数据
        chunk_size (int, optional): 分片发送大小，默认1024字节

    Raises:
        RuntimeError: 当socket发送失败时抛出
    """
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:total_sent + chunk_size])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tunned_model = AutoModel.from_pretrained("../../data/models/tunning0")
# model.model = tunned_model
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
        hidden_states, state, input_ids = client_model.sender_step(input_ids, idx)
        if state == -2: # 得到了多余的token
            continue    # 不做传输，继续生成直到得到正确的token
        # 得到了正确的token,idx+1
        idx += 1
        if state < 0 and state != -2:   # 发送出现问题或完毕，将此次内容传输后停止发送
            # send end signal
            data_tuple = pickle.dumps((hidden_states, state))
            send_large_data(client_socket, data_tuple)
            break
        
        data_tuple = pickle.dumps((hidden_states, state))
        send_large_data(client_socket, data_tuple)

        # 接收响应
        response = client_socket.recv(1024)
client_socket.close()