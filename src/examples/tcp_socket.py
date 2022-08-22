"""
Simple networking example sending messages over TCP using sockets.

Client/server portions of this code are based on this example:
<https://www.geeksforgeeks.org/python-program-that-sends-and-recieves-message-from-client/>

Usage:

```
python src/examples/tcp_socket.py
```
"""

import socket
from multiprocessing import Process


def server(ip, port):
    """
    TCP/IP server program that sends messages to client.
    """

    # create a raw socket at server side using TCP / IP protocol.
    # A socket instance is passed two parameters:
    # 1. AF_INET refers to the address family ipv4.
    # 2. SOCK_STREAM means connection-oriented TCP protocol.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket with server and port number
    s.bind((ip, port))

    # allow maximum 1 connection to the socket
    s.listen(1)

    # wait till a client accept connection
    print('server is waiting for client')
    c, addr = s.accept()

    # display client address
    print("CONNECTION FROM:", str(addr))

    # encoding into binary string then send to client
    # `b'...'` is a sequence of octets (int 0-255)
    c.send(b'Hello!')

    # change encoding to utf-8 then send to client
    # Uses one to four one-byte (8-bit) code units
    c.send("Bye...".encode())

    # disconnect the server
    c.close()


def client(server_ip, server_port, buf_size):
    """
    TCP/IP client program that receives messages from server.
    """
    # create a socket at client side using TCP / IP protocol
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # connect it to server and port number on local computer.
    s.connect((server_ip, server_port))

    # receive message string from server, at a time 1024 B
    msg = s.recv(buf_size)

    # repeat as long as message string are not empty
    while msg:
        print('Client received: ', msg.decode())
        msg = s.recv(buf_size)

    # disconnect the client
    s.close()


if __name__ == '__main__':
    HOST_IP, PORT = '127.0.0.1', 5000
    p = Process(target=server, args=(HOST_IP, PORT))
    q = Process(target=client, args=(HOST_IP, PORT, 1024))
    p.start()
    q.start()
    p.join()
