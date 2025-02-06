from kortex_api.autogen.messages import Session_pb2
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager


class DeviceConnection(object):
    IP = "192.168.1.10"
    TCP_PORTS = [10000, 10001]
    UDP_PORTS = [10002, 10003]

    @staticmethod
    def createTcpConnection(ip=IP, port=TCP_PORTS[0]):
        return DeviceConnection(ip=ip, port=port)

    @staticmethod
    def createUdpConnection(ip=IP, port=UDP_PORTS[0]):
        return DeviceConnection(ip=ip, port=port)

    def __init__(self, ip=IP, port=10000, credentials=("admin", "admin")):
        self.ip = ip
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = (
            UDPTransport() if port in DeviceConnection.UDP_PORTS else TCPTransport()
        )
        self.router = RouterClient(self.transport, self.tcp_error_callback)

    def __enter__(self):
        self.transport.connect(self.ip, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (ms)
            session_info.connection_inactivity_timeout = 2000  # (ms)
            self.session_manager = SessionManager(
                self.router, self.connection_timeout_callback
            )
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager != None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()

    def tcp_error_callback(self, error):
        print(f"[tcp client {self.ip}:{self.port}] error: {error}")

    def connection_timeout_callback(self):
        print(f"[tcp client {self.ip}:{self.port}] connection timeout")
