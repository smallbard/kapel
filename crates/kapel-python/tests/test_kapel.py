"""The Python module on SmolLM2-135M-Instruct, on the processor.

Runs with pytest or ``python -m unittest`` from this directory. The model is
read from the cache of the Hugging Face Hub (``$HF_HOME/hub``, else
``~/.cache/huggingface/hub``), or from ``$KAPEL_TEST_MODEL``; without it, the
tests that need it are skipped, or fail with ``KAPEL_REQUIRE_PARITY=1``.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import kapel


def find_model():
    if os.environ.get("KAPEL_TEST_MODEL"):
        return os.environ["KAPEL_TEST_MODEL"]
    home = os.environ.get("HF_HOME")
    hub = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = hub / "models--HuggingFaceTB--SmolLM2-135M-Instruct" / "snapshots"
    for snapshot in sorted(snapshots.glob("*")):
        if (snapshot / "model.safetensors").exists() and (snapshot / "tokenizer.json").exists():
            return str(snapshot)
    if os.environ.get("KAPEL_REQUIRE_PARITY") == "1":
        raise RuntimeError("SmolLM2-135M-Instruct is missing, and KAPEL_REQUIRE_PARITY=1")
    return None


MODEL = find_model()
needs_model = unittest.skipUnless(MODEL, "SmolLM2-135M-Instruct is not in the cache of the Hub")
QUESTION = [{"role": "user", "content": "What is the capital of France?"}]


class WithoutModel(unittest.TestCase):
    def test_the_version_is_a_string(self):
        self.assertRegex(kapel.__version__, r"^\d+\.\d+\.\d+")

    def test_a_bad_option_is_a_value_error(self):
        with self.assertRaisesRegex(ValueError, "gpu"):
            kapel.Model("anything", device="gpu")
        with self.assertRaisesRegex(ValueError, "int3"):
            kapel.Model("anything", quantize="int3")

    def test_a_model_that_cannot_be_read_is_a_kapel_error(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(kapel.KapelError):
                kapel.Model(empty)


@needs_model
class WithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = kapel.Model(MODEL)

    def test_tokens_come_back_as_text(self):
        ids = self.model.tokenize("Hello, world!", add_special_tokens=False)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        self.assertEqual(self.model.detokenize(ids), "Hello, world!")

    def test_the_chat_template_is_applied(self):
        self.assertTrue(self.model.has_chat_template)
        ids = self.model.chat_prompt(QUESTION)
        text = self.model.detokenize(ids)
        self.assertIn("What is the capital of France?", text)
        self.assertTrue(text.rstrip().endswith("assistant"))
        with self.assertRaises(ValueError):
            self.model.chat_prompt([{"role": "user"}])

    def test_greedy_answers_are_the_same_whole_or_streamed(self):
        whole = self.model.chat(QUESTION, max_tokens=24, temperature=0)
        self.assertIn("Paris", whole)
        pieces = list(self.model.chat(QUESTION, max_tokens=24, temperature=0, stream=True))
        self.assertGreater(len(pieces), 1)
        self.assertEqual("".join(pieces), whole)

    def test_a_prompt_is_a_text_or_its_tokens(self):
        text = self.model.generate("The sky is", max_tokens=8, temperature=0)
        ids = self.model.tokenize("The sky is")
        self.assertEqual(self.model.generate(ids, max_tokens=8, temperature=0), text)
        with self.assertRaises(ValueError):
            self.model.generate(3.5)

    def test_max_tokens_and_stop_end_the_answer(self):
        short = self.model.generate("One, two, three,", max_tokens=4, temperature=0)
        self.assertLessEqual(len(self.model.tokenize(short, add_special_tokens=False)), 4)
        long = self.model.generate("One, two, three,", max_tokens=32, temperature=0)
        stop = long.strip().split()[1]
        stopped = self.model.generate("One, two, three,", max_tokens=32, temperature=0, stop=[stop])
        self.assertNotIn(stop, stopped)
        self.assertTrue(long.startswith(stopped))

    def test_the_same_seed_gives_the_same_draws(self):
        a = self.model.generate("Once upon a time", max_tokens=16, temperature=1.0, seed=7)
        b = self.model.generate("Once upon a time", max_tokens=16, temperature=1.0, seed=7)
        self.assertEqual(a, b)

    def test_a_stream_left_early_stops(self):
        stream = self.model.generate("Once upon a time", max_tokens=200, stream=True)
        first = next(iter(stream))
        self.assertIsInstance(first, str)
        del stream
        # The model is still usable.
        self.assertTrue(self.model.generate("Hi", max_tokens=2))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@needs_model
@unittest.skipIf(sys.platform == "win32", "SIGINT")
class Serving(unittest.TestCase):
    def test_the_server_answers_until_ctrl_c(self):
        port = free_port()
        code = (
            "import kapel, sys; "
            f"kapel.serve(sys.argv[1], port={port}, cache_memory=0.25, "
            "served_model_name='smollm2')"
        )
        server = subprocess.Popen(
            [sys.executable, "-c", code, MODEL],
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(300):
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as r:
                        if r.status == 200:
                            break
                except OSError:
                    time.sleep(0.1)
            else:
                self.fail("the server did not start")
            body = json.dumps(
                {"model": "smollm2", "prompt": "The sky is", "max_tokens": 8, "temperature": 0}
            ).encode()
            request = urllib.request.Request(
                base + "/v1/completions", body, {"content-type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=60) as r:
                answer = json.load(r)
            self.assertEqual(answer["model"], "smollm2")
            self.assertEqual(answer["usage"]["completion_tokens"], 8)
            server.send_signal(signal.SIGINT)
            server.wait(timeout=30)
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()
            server.stderr.close()


if __name__ == "__main__":
    unittest.main()
