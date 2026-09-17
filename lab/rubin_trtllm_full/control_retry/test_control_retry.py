#!/usr/bin/env python3
"""Exercise the actual patched GET function against a real local HTTP server.

Only extract that function to avoid importing unrelated GPU/Ray dependencies.
Run with --source pointing at the candidate miles/utils/http_utils.py.
"""
import argparse
import ast
import asyncio
import json
import logging
from pathlib import Path
import unittest

import httpx


class ControlGetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'get')
        self.client = httpx.AsyncClient(trust_env=False)
        namespace = {'httpx': httpx, 'asyncio': asyncio, '_http_client': self.client,
                     'logger': logging.getLogger('control-retry-test')}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.get = namespace['get']
        self.requests = []
        self.mode = 'recover'
        self.server = await asyncio.start_server(self.respond, '127.0.0.1', 0)
        self.url = 'http://127.0.0.1:%d/list_workers' % self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.client.aclose()
        self.server.close()
        await self.server.wait_closed()

    async def respond(self, reader, writer):
        try:
            request = await reader.readuntil(b'\r\n\r\n')
            self.requests.append(request.split(b'\r\n', 1)[0])
            if self.mode == 'disconnect' or (self.mode == 'recover' and len(self.requests) == 1):
                return
            body = b'{"urls":["http://worker:30000"]}'
            status = b'200 OK'
            if self.mode == 'status':
                status = b'503 Service Unavailable'
            if self.mode == 'json':
                body = b'not json'
            writer.write(b'HTTP/1.1 ' + status + b'\r\nContent-Length: ' + str(len(body)).encode()
                         + b'\r\nConnection: close\r\n\r\n' + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def test_actual_disconnect_recovers(self):
        result = await self.get(self.url, transport_retries=2, request_timeout=1.0)
        self.assertEqual(result, {'urls': ['http://worker:30000']})
        self.assertEqual(self.requests, [b'GET /list_workers HTTP/1.1'] * 2)

    async def test_persistent_disconnect_stops_after_three_requests(self):
        self.mode = 'disconnect'
        with self.assertRaises(httpx.RemoteProtocolError):
            await self.get(self.url, transport_retries=2, request_timeout=1.0)
        self.assertEqual(len(self.requests), 3)

    async def test_default_call_does_not_retry(self):
        self.mode = 'disconnect'
        with self.assertRaises(httpx.RemoteProtocolError):
            await self.get(self.url)
        self.assertEqual(len(self.requests), 1)

    async def test_http_errors_are_not_retried(self):
        self.mode = 'status'
        with self.assertRaises(httpx.HTTPStatusError):
            await self.get(self.url, transport_retries=2, request_timeout=1.0)
        self.assertEqual(len(self.requests), 1)

    async def test_invalid_json_is_not_retried(self):
        self.mode = 'json'
        with self.assertRaises(json.JSONDecodeError):
            await self.get(self.url, transport_retries=2, request_timeout=1.0)
        self.assertEqual(len(self.requests), 1)

    async def test_invalid_retry_budget_sends_no_request(self):
        for value in (-1, 3, True):
            with self.assertRaises(ValueError):
                await self.get(self.url, transport_retries=value)
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=['test_control_retry.py', *remaining])
