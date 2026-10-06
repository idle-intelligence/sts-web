#!/usr/bin/env python3
"""Static file server for the built site, stdlib only.

Serves `_site` by default - run `scripts/build.sh` first - so the pages see
the same assembled tree GitHub Pages deploys.

Usage:
    python3 scripts/serve.py [--dir DIR] [--port PORT]
"""
import argparse
import http.server
import os
import socketserver
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="_site")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    handler = lambda *a, **kw: Handler(*a, directory=root, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {root} on http://{args.bind}:{args.port} (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
