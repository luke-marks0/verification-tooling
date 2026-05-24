from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from modules.attestation.proverdet.traffic_publisher import TrafficPublisher


class _RecordingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        # Append to the bound server's class attribute. The handler is
        # instantiated per-request; we want shared state across requests.
        srv = self.server
        if not hasattr(srv, "received_bodies"):
            srv.received_bodies = []
        srv.received_bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _spawn_recording_server() -> tuple[_RecordingServer, str]:
    server = _RecordingServer(("127.0.0.1", 0), _Handler)
    server.received_bodies = []  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address[0], server.server_address[1]
    return server, f"http://{host}:{port}"


class TestTrafficPublisher(unittest.TestCase):
    def test_publisher_buffers_and_flushes(self) -> None:
        server, url = _spawn_recording_server()
        try:
            pub = TrafficPublisher(verifier_url=url, max_batch_bytes=4096)
            pub.start()
            for i in range(100):
                pub.publish(f"frame-{i:04d}-".encode() + b"x" * 1024)
            pub.stop(timeout=10.0)
            received = b"".join(server.received_bodies)  # type: ignore[attr-defined]
            expected = b"".join(f"frame-{i:04d}-".encode() + b"x" * 1024 for i in range(100))
            self.assertEqual(received, expected)
        finally:
            server.shutdown()
            server.server_close()

    def test_publisher_does_not_drop_on_stop(self) -> None:
        server, url = _spawn_recording_server()
        try:
            pub = TrafficPublisher(verifier_url=url, max_batch_bytes=4096)
            pub.start()
            for i in range(5):
                pub.publish(f"f{i}".encode())
            pub.stop(timeout=5.0)
            got = b"".join(server.received_bodies)  # type: ignore[attr-defined]
            self.assertEqual(got, b"f0f1f2f3f4")
        finally:
            server.shutdown()
            server.server_close()

    def test_publisher_no_op_when_no_frames_published(self) -> None:
        server, url = _spawn_recording_server()
        try:
            pub = TrafficPublisher(verifier_url=url)
            pub.start()
            pub.stop(timeout=5.0)
            self.assertEqual(server.received_bodies, [])  # type: ignore[attr-defined]
        finally:
            server.shutdown()
            server.server_close()

    def test_publisher_safe_to_stop_without_start(self) -> None:
        # Don't crash if the user calls stop() before start().
        pub = TrafficPublisher(verifier_url="http://127.0.0.1:1")
        pub.stop(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
