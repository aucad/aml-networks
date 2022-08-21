"""
Simple networking example using sockets

Usage:

```
python src/examples/sockets.py
```
"""

import socket
from multiprocessing import Process

HOST = '127.0.0.1'
PORT = 5000


def server():
    """
    TCP/IP server program that sends messages to client.
    """

    # create a socket at server side using TCP / IP protocol.
    # A socket instance is passed two parameters:
    # 1. AF_INET refers to the address family ipv4.
    # 2. SOCK_STREAM means connection-oriented TCP protocol.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket with server and port number
    s.bind(('', PORT))

    # allow maximum 1 connection to the socket
    s.listen(1)

    # wait till a client accept connection
    print('server is waiting for client')
    c, addr = s.accept()

    # display client address
    print("CONNECTION FROM:", str(addr))

    # encoding into binary string then send to client
    # `b'...'` is a sequence of octets (int 0-255)
    c.send(b"Hello, how are you?")

    # change encoding to utf-8 then send to client
    # Uses one to four one-byte (8-bit) code units
    c.send("Bye..............".encode())

    # disconnect the server
    c.close()


def client():
    """
    TCP/IP server program that receive message from server.
    """
    # create a socket at client side using TCP / IP protocol
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # connect it to server and port number on local computer.
    s.connect((HOST, PORT))

    # receive message string from server, at a time 1024 B
    msg = s.recv(1024)

    # repeat as long as message string are not empty
    while msg:
        print('Received:' + msg.decode())
        msg = s.recv(1024)

    # disconnect the client
    s.close()


if __name__ == '__main__':
    p = Process(target=server)
    p.start()
    q = Process(target=client)
    q.start()
    p.join()
