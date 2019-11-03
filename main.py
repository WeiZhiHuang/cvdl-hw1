import os
import sys
import cv2
import PyQt5
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtTest
from Ui_main import Ui_Form as Ui_main
from Ui_ar import Ui_Form as Ui_ar


BOARD_SIZE = (11, 8)


def drawPyramid(img, points):
    lines = [(0, 1), (0, 3), (1, 2), (2, 3)]
    for p1, p2 in lines:
        img = cv2.line(img, tuple(points[p1].ravel()), tuple(
            points[p2].ravel()), (0, 0, 255), 10)

    top = np.divide(
        np.array([points[4], points[5], points[6], points[7]]).sum(axis=0), 4)
    for i in range(0, 4):
        img = cv2.line(img, tuple(points[i].ravel()), tuple(
            top.ravel()), (0, 0, 255), 10)
    return img


def augmentedReality():
    imgs = []
    objPoints = []
    imgPoints = []
    objPoint = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objPoint[:, :2] = np.mgrid[0:BOARD_SIZE[0],
                               0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objPoint = objPoint[::-1]

    for i in range(1, 6):
        img = cv2.imread(os.path.join('images', 'CameraCalibration', str(
            i) + '.bmp').replace('\\', '/'))  # error with UTF-8 characters
        isFound, corners = cv2.findChessboardCorners(img, BOARD_SIZE)

        if isFound:
            imgs.append(img)
            objPoints.append(objPoint)
            imgPoints.append(corners)

    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objPoints, imgPoints, BOARD_SIZE, None, None)
    axis = np.float32([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0],
                       [-1, -1, -2], [-1, 1, -2], [1, 1, -2], [1, -1, -2]]).reshape(-1, 3)

    ArWidget.show()
    for i in range(0, 5):
        imgPoints, _ = cv2.projectPoints(
            axis, rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        drawPyramid(imgs[i], imgPoints)
        qImg = QtGui.QImage(imgs[i].data,
                            imgs[i].shape[1],
                            imgs[i].shape[0],
                            imgs[i].strides[0],
                            QtGui.QImage.Format_RGB888).rgbSwapped()
        ui_ar.label.setPixmap(QtGui.QPixmap.fromImage(qImg).scaled(512, 512))
        QtTest.QTest.qWait(500)
    ArWidget.close()


if __name__ == '__main__':
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
        os.path.dirname(PyQt5.__file__), 'Qt', 'plugins', 'platforms')
    app = QtWidgets.QApplication(sys.argv)
    Widget = QtWidgets.QWidget()
    ui = Ui_main()
    ui.setupUi(Widget)

    ArWidget = QtWidgets.QWidget()
    ui_ar = Ui_ar()
    ui_ar.setupUi(ArWidget)

    ui.pushButton_5.clicked.connect(augmentedReality)
    # ui.pushButton_9.clicked.connect()
    # ui.pushButton_10.clicked.connect()
    # ui.pushButton_11.clicked.connect()
    # ui.pushButton_12.clicked.connect()
    # ui.pushButton_13.clicked.connect()

    Widget.show()
    sys.exit(app.exec_())
