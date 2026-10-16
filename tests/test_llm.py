import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from meetassist.llm import LLMClient, LLMError, _endpoint

class H(BaseHTTPRequestHandler):
    seen = []
    mode = "ok"
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        H.seen.append((self.path, self.headers.get("Authorization"), body))
        if H.mode == "401":
            self.send_response(401); self.end_headers(); self.wfile.write(b'{"error":"bad key"}'); return
        user = body["messages"][1]["content"]; sysmsg = body["messages"][0]["content"]
        if "JSON array" in sysmsg:
            n = len([l for l in user.splitlines() if l.strip()])
            out = json.dumps([f"FIXED {i+1}" for i in range(n)])
            if H.mode == "fenced": out = "```json\n" + out + "\n```"
            if H.mode == "short": out = json.dumps(["only one"])
        else:
            out = "## Summary\n- ok"
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": out}}]}).encode())

@pytest.fixture(scope="module")
def server():
    s = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"; s.shutdown()

def test_endpoint_forms():
    assert _endpoint("https://h/v1") == "https://h/v1/chat/completions"
    assert _endpoint("https://h") == "https://h/v1/chat/completions"
    assert _endpoint("https://h/v1/chat/completions/") == "https://h/v1/chat/completions"
    assert _endpoint("https://h/api/v2") == "https://h/api/v2/chat/completions"

def test_clean_lines_batches_and_key_in_header_only(server):
    H.mode = "ok"; H.seen.clear()
    c = LLMClient(server + "/v1", "SECRET", "m")
    out = c.clean_lines([f"line {i} " + "x" * 400 for i in range(30)])
    assert out == [f"FIXED {i+1}" for i in range(len(out))] or len(out) == 30
    assert len(H.seen) > 1                                       # 30*400 chars -> several batches
    assert all(a == "Bearer SECRET" for _, a, _ in H.seen)
    assert all("SECRET" not in json.dumps(b) for _, _, b in H.seen)
    assert H.seen[0][0] == "/v1/chat/completions"

def test_fenced_json_accepted(server):
    H.mode = "fenced"; c = LLMClient(server, "k")
    assert c.clean_lines(["a", "b"]) == ["FIXED 1", "FIXED 2"]

def test_length_mismatch_keeps_raw(server):
    H.mode = "short"; c = LLMClient(server, "k")
    assert c.clean_lines(["a", "b", "c"]) == ["a", "b", "c"]     # never lose/misalign lines

def test_auth_error_not_retried(server):
    H.mode = "401"; H.seen.clear(); c = LLMClient(server, "k")
    with pytest.raises(LLMError): c.ping()
    assert len(H.seen) == 1

def test_summary(server):
    H.mode = "ok"; assert "Summary" in LLMClient(server, "k").summarize("[00:00:01] [You] hello\n")

def test_unconfigured():
    with pytest.raises(LLMError): LLMClient("", "").ping()
