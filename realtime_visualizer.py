import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from gui_window import CSIMonitorWindow

if __name__ == '__main__':
    pg.setConfigOptions(antialias=True, background='k')
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CSIMonitorWindow()
    window.show()
    sys.exit(app.exec())
