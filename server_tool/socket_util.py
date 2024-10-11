import time
import socket
import pickle
import threading
import queue
import logging

BUF_SIZ = 4096


def establish_socket(host, port, timeout=10):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            s.connect((host, port))
            return s
        except ConnectionRefusedError:
            time.sleep(0.01)
    raise TimeoutError("Could not establish connection")


def request(data, host, port):
    with establish_socket(host, port) as s:
        pickled_data = pickle.dumps(data)
        pickled_data_length = len(pickled_data)
        s.sendall(pickled_data_length.to_bytes(4, 'big'))
        s.sendall(pickled_data)

        data_length = int.from_bytes(s.recv(4), 'big')
        received_data = bytearray()
        while len(received_data) < data_length:
            packet = s.recv(min(BUF_SIZ, data_length - len(received_data)))
            if not packet:
                break
            received_data.extend(packet)
        unpickled_data = pickle.loads(received_data)
    return unpickled_data


def send(data, host, port):
    with establish_socket(host, port) as s:
        pickled_data = pickle.dumps(data)
        pickled_data_length = len(pickled_data)
        s.sendall(pickled_data_length.to_bytes(4, 'big'))
        s.sendall(pickled_data)


def terminate(host, port):
    with establish_socket(host, port) as s:
        pickled_data = pickle.dumps(None)
        pickled_data_length = len(pickled_data)
        s.sendall(pickled_data_length.to_bytes(4, 'big'))
        s.sendall(pickled_data)


def bind(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    return s


def conn_recv(conn):
    data_length = int.from_bytes(conn.recv(4), 'big')
    received_data = bytearray()
    while len(received_data) < data_length:
        packet = conn.recv(min(BUF_SIZ, data_length - len(received_data)))
        if not packet:
            break
        received_data.extend(packet)
    unpickled_data = pickle.loads(received_data)
    return unpickled_data


def conn_resp(conn, data):
    pickled_data = pickle.dumps(data)
    pickled_data_length = len(pickled_data)
    conn.sendall(pickled_data_length.to_bytes(4, 'big'))
    conn.sendall(pickled_data)


class EventLoop:

    def __init__(self, host, port):
        self.s = bind(host, port)
        self.s.settimeout(0.1)

        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

        self.queue = queue.Queue()
        self.logger = logging.getLogger(self.__class__.__name__)

    def close(self):
        self.running = False
        self.thread.join()
        self.s.close()

    def __del__(self):
        if self.running:
            self.close()

    def loop(self):
        while self.running:
            try:
                conn, addr = self.s.accept()
                self.queue.put((conn, addr))
            except socket.timeout:
                pass
            except Exception as e:
                self.logger.error(e)

    def poll(self):
        try:
            conn, addr = self.queue.get_nowait()
            return conn, addr
        except queue.Empty:
            return None
