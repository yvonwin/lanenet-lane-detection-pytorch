'''
Author: your name
Date: 2021-10-13 14:52:33
LastEditTime: 2021-10-14 14:19:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CPHONE/opencv-socket_zhongche_test/serve.py
'''
import socketserver
import struct
import threading


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
  

    def setup(self):
        ip = self.client_address[0].strip()     # 获取客户端的ip
        port = self.client_address[1]           # 获取客户端的port
        print(ip+":"+str(port)+" is connect!")

    def handle(self):
        
        while True:
            self.data = self.request.recv(1024).strip()
            # cur_thread = threading.current_thread()
            # print(cur_thread)
            # print(self.data)
            news = struct.unpack('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', self.data)
            #print(news)
            # print(type(news))  # is tuple
            # 取出数据
            status  = news[0]
            if status:
                num_boxes = news[1]
                l = list(news[2:])
                # 拆分boxes
                n = 4
                boxes=[l[i:i + n] for i in range(0, len(l), n)]
                print("receive boxes成功, boxes为",boxes)

    def finish(self):
        print('client is disconnected')

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass
        
if __name__ == "__main__":
    HOST, PORT = "127.0.0.1", 8200
    print("listening")
   # server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    #ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # # # Exit the server thread when the main thread terminates
    server_thread.daemon = False
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)
    # server.serve_forever()