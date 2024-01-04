# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setCoordinates_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from human_counting import Algorithm_Count 
from set_coordinates import ClickPoints 


class Ui_setCoordinatesWindow(object):
    def __init__(self, selected_file): 
        self.selected_file = selected_file
        self.area1 = []
        self.area2 = []

    def setCoordinates_area1(self):
        coord = ClickPoints(self.selected_file, self.area2)
        self.area1 = coord.run()
        if not self.area1:
            self.show_no_coordinates_popup(True)
        elif len(self.area1) < 4:
            self.show_incomplete_popup(True)
        else:
            self.area1 = self.area1
            self.Area1Lbl.setText(str(self.area1))

    def setCoordinates_area2(self):
        coord = ClickPoints(self.selected_file, self.area1)
        self.area2 = coord.run()
        if not self.area2:
            self.show_no_coordinates_popup(False)
        elif len(self.area2) < 4:
            self.show_incomplete_popup(False)
        else:
            self.area2 = self.area2
            self.Area2Lbl.setText(str(self.area2))

    def process_video(self, setCoordinatesWindow):
        if not self.area1 and not self.area2:
            self.popup_submitBtn(True)
        elif len(self.area1) < 4 or len(self.area2) < 4:
            self.popup_submitBtn(False)
        else:
            setCoordinatesWindow.close()
            algo = Algorithm_Count(self.area1, self.area2)
            algo.counting(self.selected_file)

    def popup_submitBtn(self, no_coords):
        msg = QMessageBox()
        if no_coords:
            msg.setWindowTitle("Critical")
            msg.setText("No coordinates")
            msg.setInformativeText("Please input coordinates")
            msg.setIcon(QMessageBox.Critical)
        else:
            msg.setWindowTitle("Warning")
            msg.setText("Incomplete coordinates")
            msg.setInformativeText("Please complete coordinates")
            msg.setIcon(QMessageBox.Warning)

        result = msg.exec_()

    def show_incomplete_popup(self, bool_popup):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText("Incomplete coordinates")
        msg.setInformativeText("Please complete coordinates")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Cancel)

        if bool_popup:
            # Connect the finished signal to handle both Retry and Cancel cases
            msg.finished.connect(self.handle_popup_finished_area1)
        else:
            msg.finished.connect(self.handle_popup_finished_area2)

        result = msg.exec_()

    def show_no_coordinates_popup(self, bool_popup):
        msg = QMessageBox()
        msg.setWindowTitle("Critical")
        msg.setText("No coordinates")
        msg.setInformativeText("Please input coordinates")
        msg.setIcon(QMessageBox.Critical)
        msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Cancel)


        if bool_popup:
            # Connect the finished signal to handle both Retry and Cancel cases
            msg.finished.connect(self.handle_popup_finished_area1)
        else:
            msg.finished.connect(self.handle_popup_finished_area2)

        result = msg.exec_()
    
    def handle_popup_finished_area1(self, result):
        if result == QMessageBox.Retry:
            # Retry button clicked, call setCoordinates_area1 again
            self.setCoordinates_area1()
        elif result == QMessageBox.Cancel:
            # Cancel button or X button clicked, set self.area1 to an empty list
            self.area1 = []
            # Update the UI with the current coordinates
            self.Area1Lbl.setText("Empty list")

    def handle_popup_finished_area2(self, result):
        if result == QMessageBox.Retry:
            # Retry button clicked, call setCoordinates_area1 again
            self.setCoordinates_area2()
        elif result == QMessageBox.Cancel:
            # Cancel button or X button clicked, set self.area1 to an empty list
            self.area2 = []
            # Update the UI with the current coordinates
            self.Area2Lbl.setText("Empty list")    

    def setupUi(self, setCoordinatesWindow):
        setCoordinatesWindow.setObjectName("setCoordinatesWindow")
        setCoordinatesWindow.resize(700, 400)
        setCoordinatesWindow.setMinimumSize(QtCore.QSize(700, 400))
        setCoordinatesWindow.setMaximumSize(QtCore.QSize(700, 400))
        setCoordinatesWindow.setStyleSheet("background-color: rgb(22, 22, 22);")
        self.centralwidget = QtWidgets.QWidget(setCoordinatesWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 300))
        self.frame.setStyleSheet("background-color: rgb(44, 44, 44);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_10 = QtWidgets.QFrame(self.frame)
        self.frame_10.setMaximumSize(QtCore.QSize(50, 16777215))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.horizontalLayout.addWidget(self.frame_10)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setStyleSheet("")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Area1Lbl = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Area1Lbl.setFont(font)
        self.Area1Lbl.setStyleSheet("color: rgb(245, 245, 245);")
        self.Area1Lbl.setObjectName("Area1Lbl")
        self.verticalLayout_2.addWidget(self.Area1Lbl, 0, QtCore.Qt.AlignHCenter)
        self.area1Btn = QtWidgets.QPushButton(self.frame_4, clicked = lambda: self.setCoordinates_area1())
        self.area1Btn.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.area1Btn.setFont(font)
        self.area1Btn.setStyleSheet("QPushButton{\n"
"color: rgb(49, 48, 77);\n"
"    background-color: rgb(242, 242, 242);\n"
"border-radius: 30px;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(204, 204, 204);\n"
"    color: rgb(0, 0, 0);\n"
"}")
        self.area1Btn.setObjectName("area1Btn")
        self.verticalLayout_2.addWidget(self.area1Btn)
        self.frame_12 = QtWidgets.QFrame(self.frame_4)
        self.frame_12.setMaximumSize(QtCore.QSize(16777215, 30))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_2.addWidget(self.frame_12)
        self.horizontalLayout.addWidget(self.frame_4)
        self.frame_11 = QtWidgets.QFrame(self.frame)
        self.frame_11.setMaximumSize(QtCore.QSize(15, 16777215))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout.addWidget(self.frame_11)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.Area2Lbl = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Area2Lbl.setFont(font)
        self.Area2Lbl.setStyleSheet("color: rgb(245, 245, 245);")
        self.Area2Lbl.setObjectName("Area2Lbl")
        self.verticalLayout_3.addWidget(self.Area2Lbl, 0, QtCore.Qt.AlignHCenter)
        self.area2Btn = QtWidgets.QPushButton(self.frame_3, clicked = lambda: self.setCoordinates_area2())
        self.area2Btn.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.area2Btn.setFont(font)
        self.area2Btn.setStyleSheet("QPushButton{\n"
"color: rgb(49, 48, 77);\n"
"    background-color: rgb(242, 242, 242);\n"
"border-radius: 30px;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(204, 204, 204);\n"
"    color: rgb(0, 0, 0);\n"
"}")
        self.area2Btn.setObjectName("area2Btn")
        self.verticalLayout_3.addWidget(self.area2Btn)
        self.frame_13 = QtWidgets.QFrame(self.frame_3)
        self.frame_13.setMaximumSize(QtCore.QSize(16777215, 30))
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_3.addWidget(self.frame_13)
        self.horizontalLayout.addWidget(self.frame_3)
        self.frame_9 = QtWidgets.QFrame(self.frame)
        self.frame_9.setMaximumSize(QtCore.QSize(50, 16777215))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.horizontalLayout.addWidget(self.frame_9)
        self.verticalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setMaximumSize(QtCore.QSize(60, 16777215))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_2.addWidget(self.frame_6)
        self.submitBtn = QtWidgets.QPushButton(self.frame_5, clicked = lambda: self.process_video(setCoordinatesWindow))
        self.submitBtn.setMaximumSize(QtCore.QSize(500, 200))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.submitBtn.setFont(font)
        self.submitBtn.setStyleSheet("QPushButton{\n"
"color: rgb(49, 48, 77);\n"
"    background-color: rgb(242, 242, 242);\n"
"border-radius: 15px;\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(124, 198, 57);\n"
"color: rgb(203, 228, 222);\n"
"}")
        self.submitBtn.setObjectName("submitBtn")
        self.horizontalLayout_2.addWidget(self.submitBtn)
        self.frame_8 = QtWidgets.QFrame(self.frame_5)
        self.frame_8.setMaximumSize(QtCore.QSize(100, 16777215))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_2.addWidget(self.frame_8)
        self.cancelBtn = QtWidgets.QPushButton(self.frame_5, clicked = lambda: setCoordinatesWindow.close())
        self.cancelBtn.setMaximumSize(QtCore.QSize(500, 200))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.cancelBtn.setFont(font)
        self.cancelBtn.setStyleSheet("QPushButton{\n"
"color: rgb(49, 48, 77);\n"
"    background-color: rgb(242, 242, 242);\n"
"border-radius: 15px;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(240, 89, 65);\n"
"    color: rgb(203, 228, 222);\n"
"}")
        self.cancelBtn.setObjectName("cancelBtn")
        self.horizontalLayout_2.addWidget(self.cancelBtn)
        self.frame_7 = QtWidgets.QFrame(self.frame_5)
        self.frame_7.setMaximumSize(QtCore.QSize(60, 16777215))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_2.addWidget(self.frame_7)
        self.verticalLayout_4.addWidget(self.frame_5)
        self.verticalLayout.addWidget(self.frame_2)
        setCoordinatesWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(setCoordinatesWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 700, 21))
        self.menubar.setObjectName("menubar")
        setCoordinatesWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(setCoordinatesWindow)
        self.statusbar.setObjectName("statusbar")
        setCoordinatesWindow.setStatusBar(self.statusbar)

        self.retranslateUi(setCoordinatesWindow)
        QtCore.QMetaObject.connectSlotsByName(setCoordinatesWindow)

    def retranslateUi(self, setCoordinatesWindow):
        _translate = QtCore.QCoreApplication.translate
        setCoordinatesWindow.setWindowTitle(_translate("setCoordinatesWindow", "Set Coordinates"))
        self.Area1Lbl.setText(_translate("setCoordinatesWindow", "Empty List"))
        self.area1Btn.setText(_translate("setCoordinatesWindow", "Area 1 "))
        self.Area2Lbl.setText(_translate("setCoordinatesWindow", "Empty List"))
        self.area2Btn.setText(_translate("setCoordinatesWindow", "Area 2"))
        self.submitBtn.setText(_translate("setCoordinatesWindow", "Submit"))
        self.cancelBtn.setText(_translate("setCoordinatesWindow", "Cancel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    setCoordinatesWindow = QtWidgets.QMainWindow()
    ui = Ui_setCoordinatesWindow()
    ui.setupUi(setCoordinatesWindow)
    setCoordinatesWindow.show()
    sys.exit(app.exec_())
