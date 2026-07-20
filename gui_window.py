import os
import time
import queue
from typing import Optional

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from config import (
    COM_PORT, BAUD_RATE, WINDOW_SIZE, VARIANCE_THRESHOLD,
    GUI_INTERVAL_MS, GUI_QUEUE_SIZE, DUMP_QUEUE_SIZE
)
from circular_buffer import CircularBuffer
from serial_worker import SerialWorker
from dump_worker import DumpWorker


class CSIMonitorWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._gui_queue  = queue.Queue(maxsize=GUI_QUEUE_SIZE)
        self._dump_queue = queue.Queue(maxsize=DUMP_QUEUE_SIZE)

        self._sub_raw_buf  = CircularBuffer(WINDOW_SIZE)
        self._sub_filt_buf = CircularBuffer(WINDOW_SIZE)
        self._rssi_buf     = CircularBuffer(WINDOW_SIZE)
        self._var_buf      = CircularBuffer(WINDOW_SIZE)

        self._frames_rendered = 0
        self._pkts_total      = 0
        self._t_diag_ref      = time.monotonic()
        self._dump_active     = False
        self._gui_paused      = False

        self._build_ui()
        self._start_workers()
        self._start_timers()

    def _build_ui(self) -> None:
        self.setWindowTitle("WiFi CSI Presence Detection")
        self.resize(1180, 780)
        self.setStyleSheet("background-color: #0a0a0a;")

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(10, 10, 10, 8)

        # Plot 1: Amplitudo Subcarrier #30 — raw vs filtered
        self._plot1 = pg.PlotWidget(
            title="<span style='color:#aaaaaa;font-size:12px'>"
                  "Amplitudo CSI — Subcarrier #30 (abu: raw, biru: filtered Hampel+SG)</span>")
        self._plot1.setLabel('left', 'Amplitudo', color='#888888')
        self._plot1.showGrid(x=True, y=True, alpha=0.25)
        self._plot1.setYRange(0, 40)
        self._plot1.setXRange(0, WINDOW_SIZE)
        self._plot1.setMouseEnabled(x=False, y=False)
        self._curve1_raw  = self._plot1.plot(
            pen=pg.mkPen('#444466', width=1), skipFiniteCheck=True)
        self._curve1_filt = self._plot1.plot(
            pen=pg.mkPen('#3366FF', width=2), skipFiniteCheck=True)
        root.addWidget(self._plot1)

        # Plot 2: RSSI Real-Time
        self._plot2 = pg.PlotWidget(
            title="<span style='color:#aaaaaa;font-size:12px'>RSSI Real-Time (dBm)</span>")
        self._plot2.setLabel('left', 'RSSI (dBm)', color='#888888')
        self._plot2.showGrid(x=True, y=True, alpha=0.25)
        self._plot2.setYRange(-90, -20)
        self._plot2.setXRange(0, WINDOW_SIZE)
        self._plot2.setMouseEnabled(x=False, y=False)
        self._curve2 = self._plot2.plot(
            pen=pg.mkPen('#00DDAA', width=2), skipFiniteCheck=True)
        root.addWidget(self._plot2)

        # Plot 3: Moving Variance dari sinyal filtered
        self._plot3 = pg.PlotWidget(
            title="<span style='color:#aaaaaa;font-size:12px'>Moving Variance (dari sinyal filtered)</span>")
        self._plot3.setLabel('left', 'Variansi', color='#888888')
        self._plot3.setLabel('bottom', 'Paket', color='#888888')
        self._plot3.showGrid(x=True, y=True, alpha=0.25)
        self._plot3.setYRange(0, 40)
        self._plot3.setXRange(0, WINDOW_SIZE)
        self._plot3.setMouseEnabled(x=False, y=False)
        self._curve3 = self._plot3.plot(
            pen=pg.mkPen('#FFA500', width=2), skipFiniteCheck=True)
        thr_line = pg.InfiniteLine(
            pos=VARIANCE_THRESHOLD, angle=0,
            pen=pg.mkPen('#FF3333', width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            label=f'Threshold = {VARIANCE_THRESHOLD}',
            labelOpts={'color': '#FF5555', 'position': 0.92, 'fill': '#1a0000'})
        self._plot3.addItem(thr_line)
        root.addWidget(self._plot3)

        # Status LSTM
        self._status_lbl = QtWidgets.QLabel("STATUS: SINKRONISASI...")
        self._status_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(
            "font-size:18px;font-weight:bold;color:#888888;"
            "background:#111111;padding:6px;border-radius:5px;")
        root.addWidget(self._status_lbl)

        # Panel kontrol
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)

        self._dump_btn = QtWidgets.QPushButton("Mulai Dump")
        self._dump_btn.setFixedSize(140, 34)
        self._dump_btn.setStyleSheet(self._btn_style('#33FF33', '#002200', '#33FF33'))
        self._dump_btn.clicked.connect(self._toggle_dump)
        ctrl.addWidget(self._dump_btn)

        self._pause_btn = QtWidgets.QPushButton("Pause Grafik")
        self._pause_btn.setFixedSize(140, 34)
        self._pause_btn.setStyleSheet(self._btn_style('#FFAA00', '#221100', '#FFAA00'))
        self._pause_btn.clicked.connect(self._toggle_pause)
        ctrl.addWidget(self._pause_btn)

        self._dump_file_lbl = QtWidgets.QLabel("File dump: (belum dimulai)")
        self._dump_file_lbl.setStyleSheet(
            "font-size:11px;color:#888888;background:#0a0a0a;padding:2px;")
        ctrl.addWidget(self._dump_file_lbl, stretch=1)

        self._dump_count_lbl = QtWidgets.QLabel("Tersimpan: 0 paket")
        self._dump_count_lbl.setFixedWidth(155)
        self._dump_count_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._dump_count_lbl.setStyleSheet(
            "font-size:11px;color:#888888;background:#0a0a0a;padding:2px;")
        ctrl.addWidget(self._dump_count_lbl)
        root.addLayout(ctrl)

        self._diag_lbl = QtWidgets.QLabel(
            "GUI Queue: 0 | Dump Queue: 0 | Total: 0 | -- FPS")
        self._diag_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._diag_lbl.setStyleSheet(
            "font-size:11px;color:#444444;background:#0a0a0a;padding:2px;")
        root.addWidget(self._diag_lbl)

    @staticmethod
    def _btn_style(color: str, bg: str, border: str) -> str:
        return (f"font-size:13px;font-weight:bold;color:{color};"
                f"background:{bg};border:1px solid {border};"
                f"border-radius:4px;padding:0 12px;")

    def _toggle_dump(self) -> None:
        if not self._dump_active:
            self._dump_worker.start_dump()
        else:
            self._dump_worker.stop_dump()

    def _on_dump_started(self, filepath: str) -> None:
        self._dump_active = True
        self._dump_btn.setText("Stop Dump")
        self._dump_btn.setStyleSheet(self._btn_style('#FF4444', '#220000', '#FF4444'))
        self._dump_file_lbl.setText(
            f"{os.path.basename(filepath)}  [{os.path.abspath(filepath)}]")
        self._dump_count_lbl.setText("Tersimpan: 0 paket")

    def _on_dump_stopped(self, count: int) -> None:
        self._dump_active = False
        self._dump_btn.setText("Mulai Dump")
        self._dump_btn.setStyleSheet(self._btn_style('#33FF33', '#002200', '#33FF33'))
        self._dump_count_lbl.setText(f"{count} paket tersimpan")

    def _on_dump_count(self, count: int) -> None:
        self._dump_count_lbl.setText(f"Tersimpan: {count} paket")

    def _toggle_pause(self) -> None:
        self._gui_paused = not self._gui_paused
        if self._gui_paused:
            self._pause_btn.setText("Resume Grafik")
            self._pause_btn.setStyleSheet(self._btn_style('#4488FF', '#001133', '#4488FF'))
        else:
            self._pause_btn.setText("Pause Grafik")
            self._pause_btn.setStyleSheet(self._btn_style('#FFAA00', '#221100', '#FFAA00'))

    def _start_workers(self) -> None:
        self._dump_worker = DumpWorker(self._dump_queue)
        self._dump_worker.dump_started.connect(self._on_dump_started)
        self._dump_worker.dump_stopped.connect(self._on_dump_stopped)
        self._dump_worker.dump_count_updated.connect(self._on_dump_count)
        self._dump_worker.start()

        self._serial_worker = SerialWorker(
            COM_PORT, BAUD_RATE, self._gui_queue, self._dump_queue)
        self._serial_worker.connection_ok.connect(self._on_serial_ok)
        self._serial_worker.connection_lost.connect(self._on_serial_lost)
        self._serial_worker.start()

    def _start_timers(self) -> None:
        self._gui_timer = QtCore.QTimer(self)
        self._gui_timer.timeout.connect(self._flush_and_render)
        self._gui_timer.start(GUI_INTERVAL_MS)

        self._diag_timer = QtCore.QTimer(self)
        self._diag_timer.timeout.connect(self._refresh_diagnostics)
        self._diag_timer.start(3000)

    def _flush_and_render(self) -> None:
        n          = 0
        last_label: Optional[int] = None
        last_ts    = None  # TAMBAHAN 1: Buat nyimpen timestamp paket terakhir

        try:
            while True:
                # Format queue dari serial_worker:
                sub30_raw, sub30_filt, rssi, var, label, _pkt, _ts, _amp = \
                    self._gui_queue.get_nowait()

                self._sub_raw_buf.append(sub30_raw)
                self._sub_filt_buf.append(sub30_filt)
                self._rssi_buf.append(rssi)
                self._var_buf.append(var)

                last_label = label
                last_ts = _ts  # TAMBAHAN 2: Simpan waktu dari ESP32
                n += 1
        except queue.Empty:
            pass

        if n == 0:
            return

        self._pkts_total += n

        if not self._gui_paused:
            self._curve1_raw.setData(self._sub_raw_buf.get_ordered())
            self._curve1_filt.setData(self._sub_filt_buf.get_ordered())
            self._curve2.setData(self._rssi_buf.get_ordered())
            self._curve3.setData(self._var_buf.get_ordered())

        if last_label is not None:
            self._update_status(last_label)
            
            # TAMBAHAN 3: Hitung selisih waktu sekarang dengan _ts
            if last_ts is not None:
                current_time = time.time()
                latency_ms = (current_time - last_ts) * 1000.0
                
                # Jangan catat kalau ngaco (misal < 0)
                if latency_ms > 0:
                    with open("latency_log.txt", "a") as f:
                        f.write(f"{latency_ms:.2f}\n")

        self._frames_rendered += 1

    def _update_status(self, label: int) -> None:
        if label == 1:
            self._status_lbl.setText("STATUS: TERDETEKSI MANUSIA (OCCUPIED)")
            self._status_lbl.setStyleSheet(
                "font-size:18px;font-weight:bold;color:#FF3333;"
                "background:#220000;padding:6px;border-radius:5px;")
        else:
            self._status_lbl.setText("STATUS: RUANGAN KOSONG (EMPTY)")
            self._status_lbl.setStyleSheet(
                "font-size:18px;font-weight:bold;color:#33FF33;"
                "background:#002200;padding:6px;border-radius:5px;")

    def _refresh_diagnostics(self) -> None:
        now      = time.monotonic()
        elapsed  = now - self._t_diag_ref
        fps      = self._frames_rendered / elapsed if elapsed > 0 else 0.0
        gui_q    = self._gui_queue.qsize()
        dump_q   = self._dump_queue.qsize()
        pause_tag = "  [PAUSED]" if self._gui_paused else ""

        txt = (f"GUI Queue: {gui_q} | Dump Queue: {dump_q} | "
               f"Total: {self._pkts_total} | {fps:.1f} FPS{pause_tag}")
        self._diag_lbl.setText(txt)

        self._frames_rendered = 0
        self._t_diag_ref      = now

    def _on_serial_ok(self) -> None:
        self._status_lbl.setText("STATUS: SERIAL ONLINE - Mensinkronisasikan...")
        self._status_lbl.setStyleSheet(
            "font-size:18px;font-weight:bold;color:#4488FF;"
            "background:#001122;padding:6px;border-radius:5px;")

    def _on_serial_lost(self, msg: str) -> None:
        self._gui_timer.stop()
        self._status_lbl.setText(f"STATUS: KONEKSI TERPUTUS - {msg}")
        self._status_lbl.setStyleSheet(
            "font-size:18px;font-weight:bold;color:#FF8800;"
            "background:#221100;padding:6px;border-radius:5px;")

    def closeEvent(self, event: QtCore.QEvent) -> None:
        self._gui_timer.stop()
        self._diag_timer.stop()
        if hasattr(self, '_serial_worker'):
            self._serial_worker.stop()
        if hasattr(self, '_dump_worker'):
            self._dump_worker.stop_thread()
        event.accept()