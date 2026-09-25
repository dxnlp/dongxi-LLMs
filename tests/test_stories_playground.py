import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import unittest

from dongxi_llms.stories_playground import validate_request
from dongxi_llms.training_monitor import LogStore, handler


class PlaygroundTests(unittest.TestCase):
    def test_bounded_request_validation(self):
        self.assertEqual(validate_request({'prompt':'A rabbit'}), ('A rabbit',128,909,.8))
        for body in ([], {}, {'prompt':' '}, {'prompt':'x'*8193},
                     {'prompt':'x','temperature':float('nan')}, {'prompt':'x','temperature':-1},
                     {'prompt':'x','max_new_tokens':1000}, {'prompt':'x','max_new_tokens':True},
                     {'prompt':'x','seed':-1}):
            with self.assertRaises(ValueError): validate_request(body)

    def test_generation_endpoint_requires_explicit_header_and_json(self):
        class Fake:
            def info(self): return dict(enabled=True)
            def generate(self, body):
                validate_request(body)
                return dict(continuation='<script>plain text only</script>')
        with tempfile.TemporaryDirectory() as directory:
            assets=Path(__file__).resolve().parents[1]/'visuals/training-monitor'
            server=ThreadingHTTPServer(('127.0.0.1',0),handler(LogStore(directory),assets,Fake()))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                headers={'Content-Type':'application/json','X-Playground-Request':'1'}
                request=Request(url+'/api/generate',data=b'{"prompt":"A rabbit"}',headers=headers)
                with urlopen(request) as response: self.assertIn('<script>',json.load(response)['continuation'])
                for extra,data,code in (({},b'{}',400),({'Origin':'https://evil.example'},b'{}',403),({'X-Playground-Request':'0'},b'{}',403)):
                    with self.assertRaises(HTTPError) as error: urlopen(Request(url+'/api/generate',data=data,headers={**headers,**extra}))
                    self.assertEqual(error.exception.code,code)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)
