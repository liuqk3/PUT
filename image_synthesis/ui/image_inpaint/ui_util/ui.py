from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import sys
import os
import PyQt5
from pathlib import Path

class Ui_Form(object):
    def setupUi(self, Form, size=(256, 256)):
        Form.setObjectName("Form")
        self.buttion_W, self.buttion_H = 90, 30 # 107, 37
        self.pushButton = QPushButton(Form)
        self.pushButton.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QPushButton(Form)
        self.pushButton_5.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QPushButton(Form)
        self.pushButton_6.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QPushButton(Form)
        self.pushButton_7.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QPushButton(Form)
        self.pushButton_8.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QPushButton(Form)
        self.pushButton_9.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QPushButton(Form)
        self.pushButton_10.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_10.setObjectName("pushButton_10")


        self.pushButton_26 = QPushButton(Form)
        self.pushButton_26.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_27 = QPushButton(Form)
        self.pushButton_27.setFixedSize(self.buttion_W, self.buttion_H)
        self.pushButton_27.setObjectName("pushButton_27")
        # self.pushButton_28 = QPushButton(Form)
        # self.pushButton_28.setFixedSize(self.buttion_W, self.buttion_H)
        # self.pushButton_28.setObjectName("pushButton_28")
        # self.pushButton_29 = QPushButton(Form)
        # self.pushButton_29.setFixedSize(self.buttion_W, self.buttion_H)
        # self.pushButton_29.setObjectName("pushButton_29")    

        self.label0 = QLabel(Form)
        self.label0.setText('Origin Image')
        self.label0.setAlignment(Qt.AlignCenter) 
        self.label1 = QLabel(Form)
        self.label1.setText('Mask')
        self.label1.setAlignment(Qt.AlignCenter)
        self.label2 = QLabel(Form)
        self.label2.setText('Result')
        self.label2.setAlignment(Qt.AlignCenter)  

        self.graphicsView = QGraphicsView(Form)
        self.graphicsView.setFixedSize(size[0], size[1])
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QGraphicsView(Form)
        self.graphicsView_2.setFixedSize(size[0], size[1])
        self.graphicsView_2.setObjectName("graphicsView_2") 
        self.graphicsView_3 = QGraphicsView(Form)
        self.graphicsView_3.setFixedSize(size[0], size[1])
        self.graphicsView_3.setObjectName("graphicsView_3")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.edit)
        self.pushButton_2.clicked.connect(Form.open)
        self.pushButton_3.clicked.connect(Form.open_mask)
        self.pushButton_4.clicked.connect(Form.clear)
        self.pushButton_5.clicked.connect(Form.undo)
        self.pushButton_6.clicked.connect(Form.save_img)
        self.pushButton_7.clicked.connect(Form.result_previous)
        self.pushButton_8.clicked.connect(Form.result_next)
        self.pushButton_9.clicked.connect(Form.image_previous)
        self.pushButton_10.clicked.connect(Form.image_next)
        self.pushButton_26.clicked.connect(Form.increase)
        self.pushButton_27.clicked.connect(Form.decrease)
        # self.pushButton_28.clicked.connect(Form.sketch_mode)
        # self.pushButton_29.clicked.connect(Form.bg_mode)

        self.grid0 = QGridLayout()
        self.grid0.addWidget(self.pushButton_9, 0,0,1,1)
        self.grid0.addWidget(self.pushButton_10, 1,0,1,1)
        self.grid0.addWidget(self.pushButton_2, 0,1,1,1)
        self.grid0.addWidget(self.pushButton_3, 1,1,1,1)
        self.grid0.addWidget(self.pushButton_4, 0,2,1,1)
        self.grid0.addWidget(self.pushButton_5, 1,2,1,1)
        self.grid0.addWidget(self.pushButton_6, 0,3,1,1)
        self.grid0.addWidget(self.pushButton_7, 0,6,1,1)
        self.grid0.addWidget(self.pushButton_8, 1,6,1,1)
        self.grid0.addWidget(self.pushButton, 1,3,1,1)
        self.grid0.addWidget(self.pushButton_26, 0,4,1,1)
        self.grid0.addWidget(self.pushButton_27, 1,4,1,1)
        # self.grid0.addWidget(self.pushButton_28, 0,5,1,1)
        # self.grid0.addWidget(self.pushButton_29, 1,5,1,1)
        
        # self.grid2 = QHBoxLayout()
        # self.grid2.addLayout(self.AddWidgt(self.graphicsView, "Original Image"))
        # self.grid2.addLayout(self.AddWidgt(self.graphicsView_2, "Mask"))
        # self.grid2.addLayout(self.AddWidgt(self.graphicsView_3, "Result"))

        self.grid2 = QGridLayout()
        self.grid2.addWidget(self.label0, 0,0,1,1)
        self.grid2.addWidget(self.label1, 0,1,1,1)
        self.grid2.addWidget(self.label2, 0,2,1,1)
        self.grid2.addWidget(self.graphicsView, 1,0,1,1)
        self.grid2.addWidget(self.graphicsView_2, 1,1,1,1)
        self.grid2.addWidget(self.graphicsView_3, 1,2,1,1)

        mainLayout = QVBoxLayout()
        Form.setLayout(mainLayout)
        Form.resize(500, 500)

        mainLayout.addLayout(self.AddLayout(self.grid0, "Main Buttons"))
        mainLayout.addLayout(self.AddLayout(self.grid2, ""))
        
        # connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Image Inpainting"))
        self.pushButton.setText(_translate("Form", "Inpaint"))
        self.pushButton_2.setText(_translate("Form", "Open Image"))
        self.pushButton_3.setText(_translate("Form", "Open Mask"))
        self.pushButton_4.setText(_translate("Form", "Clear"))
        self.pushButton_5.setText(_translate("Form", "Undo"))
        self.pushButton_6.setText(_translate("Form", "Save Image"))
        self.pushButton_7.setText(_translate("Form", "Res Prev"))
        self.pushButton_8.setText(_translate("Form", "Res Next"))
        self.pushButton_9.setText(_translate("Form", "Img Prev"))
        self.pushButton_10.setText(_translate("Form", "Img Next"))
        self.pushButton_26.setText(_translate("Form", "+"))
        self.pushButton_27.setText(_translate("Form", "-"))
        # self.pushButton_28.setText(_translate("Form", "sketch"))
        # self.pushButton_29.setText(_translate("Form", "background"))

    def AddLayout(self, widget, title=''):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        if title != '':
            widgetBox.setTitle(title)
        widgetBox.setAlignment(Qt.AlignCenter)
        widgetBox.setLayout(widget)
        widgetLayout.addWidget(widgetBox)

        return widgetLayout


    def AddWidgt(self, widget, title):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)
        widgetBox.setAlignment(Qt.AlignCenter)
        vbox_t = QGridLayout()
        vbox_t.addWidget(widget,0,0,1,1)
        widgetBox.setLayout(vbox_t)
        widgetLayout.addWidget(widgetBox)

        return widgetLayout



if __name__ == "__main__":
    import sys
    # envpath = '/home/tzt/include/anaconda3/envs/pytorch1.8/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
        Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    )
    app = QApplication(sys.argv)
    # Form = QWidget()
    Form = QMainWindow()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

