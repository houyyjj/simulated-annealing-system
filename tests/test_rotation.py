import threading
import unittest
import xml.etree.ElementTree as ET
from http.client import HTTPConnection
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np

import energy_and_solution_ic as ic
from sa_engine_demo import simulated_annealing, simulated_annealing_steps
from serve_demo import DEFAULTS, DemoServer, Run, layout_metrics, render_layout, render_page, weighted_energy


def sample_layout():
    return np.array([
        [0, 0, 30, 10], [40, 0, 10, 10],
        [0, 40, 10, 20], [40, 40, 10, 3],
    ], dtype=float)


class RotationTests(unittest.TestCase):
    def test_ic_rotation_clips_to_bounds_without_changing_input(self):
        original = sample_layout()
        original[0, :2] = [70, 90]
        saved = original.copy()
        # IC 依序抽取擾動種類、移動模組、旋轉模組。
        with patch.object(ic.np.random, "randint", side_effect=[2, 0, 0]):
            rotated = ic.neighbor(original, 10)
        np.testing.assert_array_equal(rotated[0], [70, 70, 10, 30])
        np.testing.assert_array_equal(original, saved)
        np.testing.assert_array_equal(rotated[1:], original[1:])

    def test_metrics_and_weighted_energy_use_rotated_sizes(self):
        rotated = sample_layout()
        rotated[0, 2:] = [10, 30]
        bounds, area, wire, overlap = layout_metrics(rotated)
        self.assertEqual(bounds, (0, 0, 50, 60))
        self.assertEqual((area, wire, overlap), (3000, 1002.5, 0))
        self.assertEqual(weighted_energy(rotated, weights=(2, 3, 4)), 9007.5)

        # 所有模組疊在原點，涵蓋旋轉後的重疊懲罰。
        rotated[:, :2] = 0
        self.assertEqual(layout_metrics(rotated), ((0, 0, 10, 30), 300, 167.5, 490))
        self.assertEqual(
            weighted_energy(rotated, weights=(ic.ALPHA, ic.BETA, ic.OVERLAP_PENALTY)),
            ic.energy(rotated),
        )

    def test_svg_uses_rotated_rectangles_and_wire_centers(self):
        rotated = sample_layout()
        rotated[0, 2:] = [10, 30]
        bounds, *_ = layout_metrics(rotated)
        svg = ET.fromstring("<svg>" + render_layout(rotated, bounds) + "</svg>")
        cpu = svg.findall("g")[0]
        self.assertEqual(cpu.find("rect").get("width"), "48.00")
        self.assertEqual(cpu.find("rect").get("height"), "144.00")
        self.assertIn("方向 90°", cpu.find("title").text)
        self.assertEqual(cpu.find("text").get("x"), "60.00")
        self.assertEqual(cpu.find("text").get("y"), "420.00")
        self.assertEqual(svg.find("polyline").get("points"), "60.00,420.00 60.00,420.00 60.00,252.00")

    def test_engine_keeps_rotations_in_current_best_and_snapshots(self):
        initial = sample_layout()
        rotated = initial.copy()
        rotated[0, 2:] = [10, 30]
        energy = lambda layout: float(layout[0, 2])
        arguments = dict(energy=energy, T=1, dt=0.5, s=10, ds=0.8, k=1, m=1, lowbound=1, initial=initial)
        with patch.object(ic, "neighbor", return_value=rotated):
            frames = list(simulated_annealing_steps(**arguments))
            best, cost = simulated_annealing(**arguments)
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0][0], initial)
        np.testing.assert_array_equal(frames[1][0], rotated)
        np.testing.assert_array_equal(frames[1][2], rotated)
        np.testing.assert_array_equal(best, rotated)
        self.assertEqual(cost, 10)
        frames[1][0][0, 2] = 999
        self.assertEqual(frames[1][2][0, 2], 10)
        self.assertEqual(initial[0, 2], 30)

        with patch.object(ic, "neighbor", return_value=rotated), patch("sa_engine_demo.is_accept", return_value=False):
            rejected = list(simulated_annealing_steps(**arguments))[-1]
        np.testing.assert_array_equal(rejected[0], initial)
        np.testing.assert_array_equal(rejected[2], initial)

    def test_run_playback_preserves_rotation_and_initial_frame(self):
        initial = sample_layout()
        settings = dict(DEFAULTS, t0=1, tmin=0.75, dt=0.5, m=1, area=1, wire=0, overlap=0)
        rotated = initial.copy()
        rotated[0, 2:] = [10, 30]
        run = Run(settings, initial, started=True)
        with patch.object(ic, "neighbor", return_value=rotated), patch("sa_engine_demo.is_accept", return_value=True):
            run.advance()
        run.advance()  # 完成後不會新增幀。
        self.assertEqual(len(run.frames), 2)
        np.testing.assert_array_equal(run.frames[0].layout, initial)
        np.testing.assert_array_equal(run.frames[1].layout, rotated)
        self.assertIn("方向 90°", render_page("test", run, 1))
        self.assertIn("方向 0°", render_page("test", run, 0))

    def test_http_initial_page_form_and_next_round(self):
        with DemoServer(("127.0.0.1", 0)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = HTTPConnection(*server.server_address, timeout=5)
            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("晶片佈局", response.read().decode("utf-8"))
                run_id = next(iter(server.runs))
                form = dict(DEFAULTS, run=run_id, t0=1, tmin=0.75, dt=0.5, m=1)
                connection.request("POST", "/run", urlencode(form), {"Content-Type": "application/x-www-form-urlencoded"})
                response = connection.getresponse()
                self.assertEqual(response.status, 303)
                location = response.getheader("Location")
                response.read()
                connection.request("GET", location)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                connection.request("GET", location.replace("round=0", "round=1"))
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("已完成", response.read().decode("utf-8"))
            finally:
                connection.close()
                server.shutdown()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
