"""Optional local browser launcher; desktop production uses private stdio IPC."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
import webbrowser

from .app import AppService


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open the local photo review interface')
    parser.add_argument('--data-dir', type=Path, default=Path('.photo-dedup'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args(argv)
    service = AppService(args.data_dir)
    token = secrets.token_urlsafe(32)
    frontend = Path(__file__).parent / 'web'
    if not frontend.is_dir():
        frontend = Path(__file__).resolve().parent.parent / 'desktop' / 'dist'
    if not (frontend / 'index.html').exists():
        service.close()
        raise SystemExit('Build the interface first: npm --prefix desktop run build')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{server.server_port}', f'localhost:{server.server_port}'}

        def do_POST(self):
            if not self.valid_host() or self.path != '/api/rpc' or self.headers.get('Authorization') != f'Bearer {token}':
                self.reply(403, b'{"error":"This local session is not authorized"}')
                return
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{server.server_port}', f'http://localhost:{server.server_port}'}:
                self.reply(403, b'{"error":"Origin not allowed"}')
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 1_000_000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if body.get('version') != 1:
                    raise ValueError('Unsupported protocol')
                result = service.call(body['method'], body.get('params', {}))
                self.reply(200, json.dumps(dict(result=result)).encode())
            except Exception as error:
                self.reply(400, json.dumps(dict(error=str(error))).encode())

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, b'Invalid host', 'text/plain')
                return
            from urllib.parse import unquote, urlsplit
            relative = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
            path = (frontend / relative).resolve()
            if not path.is_relative_to(frontend.resolve()):
                self.reply(403, b'Forbidden', 'text/plain')
                return
            if not path.is_file():
                self.reply(404, b'Not found', 'text/plain')
                return
            self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    url = f'http://127.0.0.1:{server.server_port}/#{token}'
    print(f'Local review app: {url}', flush=True)
    print('Your photos remain on this computer. Press Ctrl+C to stop.', flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
