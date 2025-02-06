# Author: Jimmy Wu
# Date: September 2022

import argparse
import time
from multiprocessing.connection import Listener
from multiprocessing.pool import ThreadPool


class Server(object):
    def __init__(self, hostname="localhost", start_port=6000, n_conns=1):
        self.listeners = []
        for i in range(n_conns):
            print(f"{type(self)} Creating a listener at port {start_port + i}")
            l = Listener((hostname, start_port + i), authkey=b"secret password")
            self.listeners.append(l)

    def get_data(self, req):
        time.sleep(0.033)
        return {}

    def clean_up(self):
        print("Cleaning up")

    def run(self):
        try:
            pool = ThreadPool()
            pool.map_async(lambda l: self._run(l), self.listeners)
            pool.close()
            pool.join()
        finally:
            self.clean_up()

    def _run(self, listener):
        while True:
            # Connect to clients
            address, port = listener.address
            print(f"[{address}:{port}] waiting for connection...")
            conn = listener.accept()
            print(f"[{address}:{port}] connected.")

            while True:
                try:
                    if not conn.poll():
                        continue
                    req = conn.recv()
                    data = self.get_data(req)
                    conn.send(data)
                except (ConnectionResetError, EOFError):
                    break

            print(f"[{address}:{port}] disconnected.")


def main(args):
    Server(start_port=args.start_port, num_conns=args.num_conns).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-port", type=int, default=6000)
    parser.add_argument("--num-conns", type=int, default=1)
    main(parser.parse_args())
