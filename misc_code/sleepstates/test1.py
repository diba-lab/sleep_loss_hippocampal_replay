from PyQt5 import QtWidgets
import sys


def main():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    main = MainWindow()
    main.show()

    return main


if __name__ == "__main__":
    m = main()
