"""
Simple networking example sending messages over UDP using sockets.

Client/server portions of this code are based on this example:
<https://pythontic.com/modules/socket/udp-client-server-example>

Usage:

```
python src/examples/udp_socket.py
```
"""

import socket
from multiprocessing import Process


def server(ip, port, buffer_size):
    """
    UDP server program
    """

    # Create a datagram socket
    # 1. AF_INET refers to the address family ipv4.
    # 2. SOCK_DGRAM means UDP protocol.
    s = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)

    # bind the socket with server and port number
    s.bind((ip, port))
    print("UDP server up and listening")

    # Listen for incoming datagrams
    while True:
        byte_addr_pair = s.recvfrom(buffer_size)
        message = byte_addr_pair[0].decode()
        address = byte_addr_pair[1]

        client_msg = "Message from Client: {}".format(message)
        client_ip = "Client IP Address: {}".format(address)

        print(client_msg)
        print(client_ip)

        # Sending a reply to client
        msg = "Hello to UDP client"
        s.sendto(str.encode(msg), address)
        s.close()
        break


def client(server_ip, server_port, buf_size):
    """
    UDP client
    """
    # create a socket at client side using TCP / IP protocol
    s = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    msg = "Hello to UDP Server"

    # Send to server using created UDP socket
    s.sendto(str.encode(msg), (server_ip, server_port))

    resp = s.recvfrom(buf_size)
    msg = "Message from Server: {} ".format(resp[0].decode())

    print(msg)
    s.close()


if __name__ == '__main__':
    HOST_IP, PORT = '127.0.0.1', 20001
    p = Process(target=server, args=(HOST_IP, PORT, 1024))
    q = Process(target=client, args=(HOST_IP, PORT, 1024))
    p.start()
    q.start()
    p.join()
