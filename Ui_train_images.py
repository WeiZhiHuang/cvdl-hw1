from PyQt5 import QtCore, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1310, 148)

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(0, 128, 30, 20)

        self.imgLabels = []
        self.labels = []
        for i in range(10):
            imgLabel = QtWidgets.QLabel(Form)
            imgLabel.setGeometry(QtCore.QRect(i * 128 + 30, 0, 128, 128))
            self.imgLabels.append(imgLabel)
            label = QtWidgets.QLabel(Form)
            label.setGeometry(QtCore.QRect(i * 128 + 30, 128, 128, 20))
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.labels.append(label)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "CVDL2019_HW1_TRAIN_IMAGES"))
        self.label.setText(_translate("Form", "Label:"))

