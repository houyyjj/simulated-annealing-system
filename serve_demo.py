# 本機網頁展示入口：接收退火參數、逐輪計算佈局，並產生 HTML 與 SVG 圖表。
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import combinations
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import numpy as np
from numpy import ndarray

import energy_and_solution_ic as problem
from sa_engine_demo import simulated_annealing_steps


# 以程式所在目錄定位網頁資源，避免從其他工作目錄啟動時找不到檔案。
PAGE = Path(__file__).resolve().parent / "page"
# t0／tmin：初始／終止溫度；dt：每輪溫度倍率；k：接受較差解的機率參數。
# s0：初始擾動範圍；ds：每輪擾動範圍倍率；m：每輪嘗試的鄰近解數量。
# area／wire／overlap：外框面積、加權線長與重疊面積的成本權重。
DEFAULTS = dict(t0=90, tmin=0.5, dt=0.9, k=1, s0=10, ds=0.8, m=100,
                area=10.0, wire=1.0, overlap=100.0)
# 表單各參數允許的範圍；dt 必須小於 1，溫度才會逐輪下降。
LIMITS = {
    "t0": (0.01, 1e6), "tmin": (0.01, 1e6), "dt": (0.01, 0.9999),
    "k": (0.01, 1e6), "s0": (0.01, max(problem.BOUND_X, problem.BOUND_Y)),
    "ds": (0.01, 1), "m": (1, 10_000),
    "area": (0, 1e6), "wire": (0, 1e6), "overlap": (0, 1e6),
}


def read_settings(form):
    # 將表單字串轉為數值，並檢查缺漏、格式與上下限。
    settings = {}
    for name, (low, high) in LIMITS.items():
        try:
            value = float(form[name])
        except (KeyError, ValueError):
            raise ValueError(f"參數 {name} 須為數字。") from None
        if not low <= value <= high:
            raise ValueError(f"參數 {name} 須介於 {low:g} 和 {high:g}。")
        settings[name] = value
    # 嘗試次數必須是整數，且初始溫度必須高於終止溫度。
    if not settings["m"].is_integer():
        raise ValueError("每輪嘗試次數須為整數。")
    settings["m"] = int(settings["m"])
    if settings["t0"] <= settings["tmin"]:
        raise ValueError("終止溫度須低於初始溫度。")
    return settings


def layout_metrics(layout):
    """依 [x, y, w, h] 計算外框、面積、加權線長與重疊面積。"""
    positions = layout[:, :2]
    sizes = layout[:, 2:4]
    lower = positions.min(axis=0)
    upper = (positions + sizes).max(axis=0)
    extent = upper - lower
    bounds = (*lower, *extent)
    area = float(np.prod(extent))

    i, j = np.triu_indices(len(layout), k=1)
    centers = positions + sizes / 2
    hpwl = np.abs(centers[i] - centers[j]).sum(axis=1)
    wire = float(np.sum(problem.W_IJ * hpwl))
    overlap_size = np.maximum(
        0.0,
        np.minimum(positions[i] + sizes[i], positions[j] + sizes[j])
        - np.maximum(positions[i], positions[j]),
    )
    overlap = float(np.prod(overlap_size, axis=1).sum())
    return bounds, area, wire, overlap


def weighted_energy(layout, *, weights):
    # 與 IC 的成本公式一致，另外支援網頁設定的三項權重。
    _, area, wire, overlap = layout_metrics(layout)
    alpha, beta, gamma = weights
    return alpha * area + beta * wire + gamma * overlap


@dataclass
class Frame:
    # 一輪畫面所需的快照：模組 [x, y, w, h]、成本、溫度與擾動範圍。
    # layout 是該輪結束時的佈局，不一定是歷史最低成本所對應的佈局。
    layout: ndarray
    current: float
    best: float
    temperature: float
    radius: float


class Run:
    # 保存一次退火執行的設定與快照，供網頁播放、暫停及回看。
    def __init__(self, settings, initial, *, started=False):
        self.settings = settings
        self.started = started
        self.rounds = 0
        # 先依降溫規則估算總輪數，並限制最多 1000 輪。
        temperature = settings["t0"]
        while temperature >= settings["tmin"]:
            self.rounds += 1
            if self.rounds > 1000:
                raise ValueError("預計超過 1000 輪，請調低降溫率或提高終止溫度。")
            temperature *= settings["dt"]

        # 用 partial 固定本次執行的成本權重，讓退火引擎只需傳入佈局。
        weights = (settings["area"], settings["wire"], settings["overlap"])
        self.energy = partial(weighted_energy, weights=weights)
        # 建立逐輪退火流程，移動、交換與旋轉由 problem.neighbor 產生。
        self.steps = simulated_annealing_steps(
            energy=self.energy,
            T=settings["t0"],
            dt=settings["dt"],
            s=settings["s0"],
            ds=settings["ds"],
            k=settings["k"],
            m=settings["m"],
            lowbound=settings["tmin"],
            initial=initial,
        )

        # 取得第 0 幀，此時尚未搜尋。
        layout, current, _, best, temperature, radius = next(self.steps)
        self.frames = [
            Frame(layout, current, best, temperature, radius)
        ]

    def advance(self):
        # 已完成所有輪次時，不再計算。
        if len(self.frames) - 1 >= self.rounds:
            return

        # 讓 sa_engine 繼續計算一輪。
        layout, current, _, best, temperature, radius = next(self.steps)

        # 保存結果，供佈局圖與成本曲線使用。
        self.frames.append(
            Frame(layout, current, best, temperature, radius)
        )


def render_layout(layout, bounds):
    # 將晶片座標縮放到 SVG 的 480 × 480 繪圖區域。
    sx, sy = 480 / problem.BOUND_X, 480 / problem.BOUND_Y

    def point(x, y):
        # 晶片座標的 y 軸向上，SVG 的 y 軸向下，因此需要反轉方向。
        return 36 + sx * x, 492 - sy * y

    def rect(x, y, width, height):
        # 晶片矩形以左下角定位，SVG 矩形則以左上角定位。
        left, top = point(x, y + height)
        return f'x="{left:.2f}" y="{top:.2f}" width="{width * sx:.2f}" height="{height * sy:.2f}"'

    # 先畫所有模組的最小外框，再依模組中心畫出有連線權重的配對。
    parts = [f'<rect class="bounding-box" {rect(*bounds)}/>']
    centers = layout[:, :2] + layout[:, 2:4] / 2
    # 配對順序與 W_IJ 一致；折線採水平加垂直的曼哈頓路徑。
    for (i, j), weight in zip(combinations(range(len(layout)), 2), problem.W_IJ):
        if weight == 0:
            continue
        x1, y1 = point(*centers[i])
        x2, y2 = point(*centers[j])
        name = f"{problem.MODULE_NAMES[i]}–{problem.MODULE_NAMES[j]}"
        parts.append(
            f'<polyline class="wire" points="{x1:.2f},{y1:.2f} {x2:.2f},{y1:.2f} {x2:.2f},{y2:.2f}">'
            f'<title>{name}，權重 {weight:g}</title></polyline>'
        )
    # 最後繪製各模組的方塊、名稱與滑鼠提示，讓模組顯示在線路上方。
    colors = ("#82afff", "#63e3ba", "#f0c081", "#d7a9ff")
    for name, (x, y, w, h), original_size, color in zip(problem.MODULE_NAMES, layout, problem.MODULES, colors):
        cx, cy = point(x + w / 2, y + h / 2)
        angle = 0 if np.array_equal((w, h), original_size) else 90
        parts.append(
            f'<g><title>{name} ({x:.2f}, {y:.2f})，{w:g} × {h:g}，方向 {angle}°</title>'
            f'<rect {rect(x, y, w, h)} rx="3" fill="#152636" stroke="{color}"/>'
            f'<text class="module-name" x="{cx:.2f}" y="{cy:.2f}" fill="{color}" '
            f'text-anchor="middle" dominant-baseline="central">{name}</text></g>'
        )
    return "".join(parts)


def render_chart(frames, rounds):
    # 依目前可見的成本資料決定縱軸範圍，保留邊距並避免範圍為零。
    low = min(frame.best for frame in frames)
    high = max(frame.current for frame in frames)
    padding = max(1, (high - low) * 0.07)
    low, high = max(0, low - padding), high + padding
    # 曲線獨立伸展填滿繪圖區；文字刻度放在 SVG 外，避免隨寬高拉伸。
    left, right, top, bottom = 3, 997, 5, 95

    def x(index):
        # 橫軸固定對應預計總輪數，播放時曲線會逐步向右延伸。
        return left + (right - left) * index / rounds

    def y(cost):
        # 成本越高，SVG 上的位置越靠上。
        return bottom - (bottom - top) * (cost - low) / (high - low)

    parts = ['<div class="chart-y-axis" aria-hidden="true">']
    for cost in (high, (low + high) / 2, low):
        label = f"{cost:.3g}" if cost >= 1e6 else f"{cost:,.1f}"
        parts.append(f'<span>{label}</span>')
    parts.append('</div><svg class="chart-plot" viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">')
    for cost in (low, (low + high) / 2, high):
        parts.append(f'<path class="chart-grid" d="M0 {y(cost):.2f}H1000" fill="none" stroke="#324559"/>')
    # 藍色實線表示當前成本，綠色虛線表示歷史最低成本。
    for key, color, pattern in (("current", "blue", ""), ("best", "mint", ' stroke-dasharray="5 4"')):
        points = " ".join(f"{x(i):.2f},{y(getattr(frame, key)):.2f}" for i, frame in enumerate(frames))
        parts.append(f'<polyline class="chart-line" points="{points}" fill="none" stroke="var(--{color})" stroke-width="2"{pattern}/>')
        # 極短線段搭配固定像素的圓形端點，放大時仍保持圓形。
        parts.append(
            f'<path class="chart-line" d="M{x(len(frames) - 1):.2f} {y(getattr(frames[-1], key)):.2f}h0.01" '
            f'stroke="var(--{color})" stroke-width="5" stroke-linecap="round"/>'
        )
    parts.append('</svg><div class="chart-x-axis" aria-hidden="true">')
    for index in sorted({0, rounds // 2, rounds}):
        parts.append(f'<span style="left:{100 * index / rounds:.3f}%">{index}</span>')
    parts.append('</div>')
    return "".join(parts)


def frame_url(run_id, index, playing=False):
    # 將執行識別碼、輪數及播放狀態編入網址，供控制按鈕與自動換頁使用。
    query = {"run": run_id, "round": index}
    if playing:
        query["play"] = 1
    return "/?" + urlencode(query)


def render_page(run_id, run, index, playing=False, *, error="", values=None):
    # 取出指定輪數的快照，計算畫面指標；到最後一輪時自動停止播放。
    frame = run.frames[index]
    bounds, area, wire, _ = layout_metrics(frame.layout)
    finished = index == run.rounds
    playing = playing and not finished
    here = escape(frame_url(run_id, index))
    next_frame = escape(frame_url(run_id, index + 1))
    resume = escape(frame_url(run_id, index + 1, True))
    reset = escape(frame_url(run_id, 0))
    # 透過 HTML 每秒換頁實現播放，不需要 JavaScript。
    refresh = f'<meta http-equiv="refresh" content="1; url={resume}">' if playing else ""

    # 根據是否開始、播放中或已完成，產生對應的控制按鈕。
    if playing:
        controls = f'<a class="button primary" id="pause" href="{here}">暫停</a>'
    elif not run.started:
        controls = '<button class="button primary" id="play" type="submit" form="parameters">開始</button>'
    elif not finished:
        controls = f'<a class="button primary" id="play" href="{resume}">繼續</a>'
    else:
        controls = '<button class="button primary" type="submit" form="parameters">重新執行</button>'
    if run.started and not playing and not finished:
        controls += f'<a class="button" id="step" href="{next_frame}">下一輪</a>'
    controls += f'<a class="button ghost" id="restart" href="{reset}">回到初始佈局</a>'

    state = "已完成 · " if finished else ""
    status = f"{state}第 {index} / {run.rounds} 輪 · T = {frame.temperature:.3g}"
    # 將文字與表單值做 HTML 跳脫，再填入範本；繪圖與按鈕則填入產生的標記。
    # 驗證失敗時優先保留使用者輸入的 values，方便直接修正。
    template = Template((PAGE / "index.html").read_text(encoding="utf-8"))
    return template.substitute(
        {key: escape(str(value), quote=True) for key, value in (values or run.settings).items()},
        refresh=refresh, layout=render_layout(frame.layout, bounds), status=status,
        playback=controls, chart=render_chart(run.frames[:index + 1], run.rounds),
        current_cost=f"{frame.current:,.2f}", best_cost=f"{frame.best:,.2f}",
        current_area=f"{area:,.2f}", current_wire=f"{wire:,.2f}", run_id=run_id,
        error=f'<p class="form-error" role="alert">{escape(error)}</p>' if error else "",
    )


class DemoServer(HTTPServer):
    # 在記憶體中保存執行紀錄；伺服器結束後這些紀錄不會保留。
    def __init__(self, address):
        super().__init__(address, DemoHandler)
        self.runs = OrderedDict()

    def store_run(self, run):
        # 以隨機識別碼存放紀錄；超過 32 筆時移除最久未使用的一筆。
        run_id = uuid4().hex
        self.runs[run_id] = run
        if len(self.runs) > 32:
            self.runs.popitem(last=False)
        return run_id

    def get_run(self, run_id):
        # 將讀取到的紀錄移至尾端，更新最近使用順序。
        run = self.runs.get(run_id)
        if run is not None:
            self.runs.move_to_end(run_id)
        return run


class DemoHandler(BaseHTTPRequestHandler):
    def send_content(self, status, content_type, body):
        # 統一傳送位元組內容、UTF-8 編碼與長度，停用快取並禁止網頁腳本。
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "script-src 'none'")
        self.end_headers()
        self.wfile.write(body)

    def send_page(self, run_id, run, index=0, playing=False, *, error="", values=None):
        # 將 HTML 編碼成位元組；表單驗證錯誤時回傳 HTTP 400。
        body = render_page(run_id, run, index, playing, error=error, values=values).encode("utf-8")
        self.send_content(400 if error else 200, "text/html", body)

    def do_GET(self):
        # GET 負責提供樣式與展示頁面，其餘路徑回傳 404。
        url = urlsplit(self.path)
        if url.path == "/styles.css":
            self.send_content(200, "text/css", (PAGE / "styles.css").read_bytes())
            return
        if url.path not in ("/", "/index.html"):
            self.send_error(404)
            return

        query = parse_qs(url.query)
        run_id = query.get("run", [""])[0]
        # 首次開啟時建立隨機初始佈局，先顯示尚未開始的第 0 幀。
        if not run_id:
            run = Run(DEFAULTS, problem.initial_solution())
            self.send_page(self.server.store_run(run), run)
            return
        run = self.server.get_run(run_id)
        if run is None:
            self.send_error(404, "Run expired. Open / to start again.")
            return
        # 輪數只能指向已有快照，或緊接在快照之後的下一輪。
        try:
            index = int(query.get("round", ["0"])[0])
        except ValueError:
            self.send_error(400)
            return
        if not 0 <= index <= min(run.rounds, len(run.frames)):
            self.send_error(404)
            return
        # 只有請求尚未計算的下一輪才推進退火，回看舊快照不會重新計算。
        if index == len(run.frames):
            run.advance()
        self.send_page(run_id, run, index, query.get("play") == ["1"])

    def do_POST(self):
        # POST /run 接收一般 HTML 表單，用提交的參數建立新執行。
        if self.path != "/run":
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/x-www-form-urlencoded":
            self.send_error(415)
            return
        # 限制請求本文大小與欄位數，保留空欄位供後續參數驗證。
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError
            fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, max_num_fields=20)
        except ValueError:
            self.send_error(400)
            return

        form = {key: values[0] for key, values in fields.items()}
        run_id = form.get("run", "")
        previous = self.server.get_run(run_id)
        # 調整參數重跑時沿用原始佈局，方便比較不同設定的結果。
        initial = previous.frames[0].layout if previous else problem.initial_solution()
        try:
            run = Run(read_settings(form), initial, started=True)
        except ValueError as error:
            # 驗證失敗時顯示原有佈局與錯誤訊息，並保留已填寫的參數。
            if previous is None:
                previous = Run(DEFAULTS, initial)
                run_id = self.server.store_run(previous)
            values = {key: form.get(key, default) for key, default in DEFAULTS.items()}
            self.send_page(run_id, previous, error=str(error), values=values)
            return
        run_id = self.server.store_run(run)
        # 建立成功後重新導向播放網址，避免重新整理頁面時重複提交表單。
        self.send_response(303)
        self.send_header("Location", frame_url(run_id, 0, True))
        self.send_header("Content-Length", "0")
        self.end_headers()


if __name__ == "__main__":
    # 從命令列讀取連接埠，只監聽本機位址；按 Ctrl+C 即可結束伺服器。
    parser = argparse.ArgumentParser(description="啟動晶片佈局實驗室")
    parser.add_argument("--port", type=int, default=8000, help="本機連接埠")
    args = parser.parse_args()
    with DemoServer(("127.0.0.1", args.port)) as server:
        print(f"晶片佈局實驗室 http://127.0.0.1:8000", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
