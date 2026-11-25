"""Direct-run startup, occupied ports, and browser launch behavior."""

import contextlib
import errno
import importlib.util
import io
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workbench_launcher", ROOT / "启动_PTDL前端.py")
workbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workbench)


class LauncherTests(unittest.TestCase):
    def test_occupied_port_serves_page_from_another_working_directory(self):
        occupied = workbench.create_server("127.0.0.1", 0)
        server = None
        thread = None
        try:
            server = workbench.create_server("127.0.0.1", occupied.server_address[1])
            self.assertNotEqual(server.server_address[1], occupied.server_address[1])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            # PyCharm can launch the same file with a different working directory.
            with contextlib.chdir(ROOT.parent):
                with urlopen(f"http://127.0.0.1:{server.server_address[1]}/", timeout=5) as response:
                    html = response.read().decode("utf8")
            self.assertIn("PTDL 飞推控协同优化工作台", html)
            self.assertIn('id="modelData"', html)
        finally:
            if server:
                if thread:
                    server.shutdown()
                server.server_close()
            if thread:
                thread.join(timeout=5)
            occupied.server_close()

    def test_main_opens_actual_port_and_closes_server(self):
        server = Mock(server_address=("127.0.0.1", 9876))
        server.serve_forever.side_effect = KeyboardInterrupt
        opened = threading.Event()
        with patch.dict(os.environ, {"PTDL_UI_HOST": "127.0.0.1", "PTDL_UI_PORT": "8765",
                                     "PTDL_UI_OPEN_BROWSER": "1"}):
            with patch.object(workbench, "create_server", return_value=server):
                with patch.object(workbench.webbrowser, "open",
                                  side_effect=lambda *a, **k: opened.set() or True) as browser:
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        workbench.main()
                        self.assertTrue(opened.wait(timeout=3))
        browser.assert_called_once_with("http://127.0.0.1:9876/", new=2)
        self.assertIn("9876", output.getvalue())
        server.server_close.assert_called_once()

    def test_auto_browser_can_be_disabled(self):
        server = Mock(server_address=("127.0.0.1", 9876))
        with patch.dict(os.environ, {"PTDL_UI_OPEN_BROWSER": "0"}):
            with patch.object(workbench, "create_server", return_value=server):
                with patch.object(workbench.webbrowser, "open") as browser:
                    with contextlib.redirect_stdout(io.StringIO()):
                        workbench.main()
        browser.assert_not_called()
        server.server_close.assert_called_once()

    def test_unexpected_bind_errors_are_reported(self):
        with patch.object(workbench, "WorkbenchHTTPServer",
                          side_effect=OSError(errno.EADDRNOTAVAIL, "Invalid host")):
            with self.assertRaises(OSError):
                workbench.create_server("192.0.2.1", 8765)


if __name__ == "__main__":
    unittest.main()
