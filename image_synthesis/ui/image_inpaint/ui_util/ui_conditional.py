# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_v3.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

stuff_classes=['things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', \
    'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', \
    'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', \
    'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', \
    'cabinet', 'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', \
    'wall', 'rug',]
thing_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', \
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', \
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', \
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', \
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', \
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', \
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', \
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

classes = stuff_classes + thing_classes
classes.sort()

num_unknown = 20

for i in range(num_unknown):
    classes.append('unknown{}'.format(i+1))


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1725, 605)
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(440, 100, 113, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(555, 100, 113, 32))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_26 = QtWidgets.QPushButton(Form)
        self.pushButton_26.setGeometry(QtCore.QRect(850, 100, 113, 32))
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(970, 100, 113, 32))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_9 = QtWidgets.QPushButton(Form)
        self.pushButton_9.setGeometry(QtCore.QRect(440, 165, 113, 32))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(115, 230, 91, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(405, 230, 60, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(665, 230, 41, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(1050, 230, 131, 16))
        self.label_4.setObjectName("label_4")
        self.pushButton_12 = QtWidgets.QPushButton(Form)
        self.pushButton_12.setGeometry(QtCore.QRect(90, 100, 113, 32))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(Form)
        self.pushButton_13.setGeometry(QtCore.QRect(90, 165, 113, 32))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(Form)
        self.pushButton_14.setGeometry(QtCore.QRect(675, 165, 113, 32))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(Form)
        self.pushButton_15.setGeometry(QtCore.QRect(675, 100, 113, 32))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_10 = QtWidgets.QPushButton(Form)
        self.pushButton_10.setGeometry(QtCore.QRect(555, 165, 113, 32))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(325, 100, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(210, 100, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_8 = QtWidgets.QPushButton(Form)
        self.pushButton_8.setGeometry(QtCore.QRect(325, 165, 113, 32))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(210, 165, 113, 32))
        self.pushButton_7.setObjectName("pushButton_7")
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(893, 160, 147, 26))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.comboBox = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.pushButton_16 = QtWidgets.QPushButton(Form)
        self.pushButton_16.setGeometry(QtCore.QRect(1090, 100, 113, 32))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_17 = QtWidgets.QPushButton(Form)
        self.pushButton_17.setGeometry(QtCore.QRect(1090, 160, 113, 32))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(Form)
        self.pushButton_18.setGeometry(QtCore.QRect(1210, 160, 113, 32))
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_19 = QtWidgets.QPushButton(Form)
        self.pushButton_19.setGeometry(QtCore.QRect(1210, 100, 113, 32))
        self.pushButton_19.setObjectName("pushButton_19")
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(25, 280, 256, 256))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(291, 280, 256, 256))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_3.setGeometry(QtCore.QRect(557, 280, 256, 256))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_4 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_4.setGeometry(QtCore.QRect(864, 281, 256, 256))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.graphicsView_5 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_5.setGeometry(QtCore.QRect(1130, 281, 256, 256))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.graphicsView_6 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_6.setGeometry(QtCore.QRect(1430, 280, 256, 256))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(1520, 230, 91, 16))
        self.label_6.setObjectName("label_6")
        self.layoutWidget1 = QtWidgets.QWidget(Form)
        self.layoutWidget1.setGeometry(QtCore.QRect(740, 30, 247, 23))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.checkBox_2 = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout.addWidget(self.checkBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout.addWidget(self.checkBox)
        self.pushButton_20 = QtWidgets.QPushButton(Form)
        self.pushButton_20.setGeometry(QtCore.QRect(1370, 160, 113, 32))
        self.pushButton_20.setObjectName("pushButton_20")
        self.pushButton_21 = QtWidgets.QPushButton(Form)
        self.pushButton_21.setGeometry(QtCore.QRect(1490, 160, 113, 32))
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_25 = QtWidgets.QPushButton(Form)
        self.pushButton_25.setGeometry(QtCore.QRect(1370, 100, 113, 32))
        self.pushButton_25.setObjectName("pushButton_25")
        self.pushButton_22 = QtWidgets.QPushButton(Form)
        self.pushButton_22.setGeometry(QtCore.QRect(1490, 100, 113, 32))
        self.pushButton_22.setObjectName("pushButton_22")
        self.pushButton_23 = QtWidgets.QPushButton(Form)
        self.pushButton_23.setGeometry(QtCore.QRect(1610, 100, 113, 32))
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_24 = QtWidgets.QPushButton(Form)
        self.pushButton_24.setGeometry(QtCore.QRect(1610, 160, 113, 32))
        self.pushButton_24.setObjectName("pushButton_24")



        self.retranslateUi(Form)

        self.pushButton_3.clicked.connect(Form.save_img)
        self.pushButton_4.clicked.connect(Form.increase)
        self.pushButton_6.clicked.connect(Form.edit_seg)
        self.pushButton_9.clicked.connect(Form.edit)
        self.pushButton_12.clicked.connect(Form.image_previous)
        self.pushButton_13.clicked.connect(Form.image_next)
        self.pushButton_14.clicked.connect(Form.result_next)
        self.pushButton_15.clicked.connect(Form.result_previous)
        self.pushButton_10.clicked.connect(Form.decrease)
        self.pushButton_2.clicked.connect(Form.clear)
        self.pushButton.clicked.connect(Form.open)
        self.pushButton_8.clicked.connect(Form.undo)
        self.pushButton_7.clicked.connect(Form.open_mask)
        self.pushButton_16.clicked.connect(Form.increase_seg)
        self.pushButton_17.clicked.connect(Form.decrease_seg)
        self.pushButton_18.clicked.connect(Form.undo_seg)
        self.pushButton_19.clicked.connect(Form.clear_seg)
        self.pushButton_26.clicked.connect(Form.open_seg_map)
        self.pushButton_20.clicked.connect(Form.undo_sketch)
        self.pushButton_21.clicked.connect(Form.clear_sketch)
        self.pushButton_22.clicked.connect(Form.increase_sketch)
        self.pushButton_23.clicked.connect(Form.decrease_sketch)
        self.pushButton_24.clicked.connect(Form.erase_sketch)
        self.pushButton_25.clicked.connect(Form.open_sketch_map)

        # self.comboBox.addItems(classes)

        # self.stuff_classes = stuff_classes
        # self.thing_classes = thing_classes
        self.num_unknown_classes = num_unknown

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Conditional Inpainting"))
        self.pushButton_3.setText(_translate("Form", "Save Image"))
        self.pushButton_4.setText(_translate("Form", "+"))
        self.pushButton_6.setText(_translate("Form", "Edit Seg Map"))
        self.pushButton_9.setText(_translate("Form", "Inpaint"))
        self.label.setText(_translate("Form", "Original Image"))
        self.label_2.setText(_translate("Form", "Mask"))
        self.label_3.setText(_translate("Form", "Result"))
        self.label_4.setText(_translate("Form", "Segmentation Map"))
        self.pushButton_12.setText(_translate("Form", "Img Prev"))
        self.pushButton_13.setText(_translate("Form", "Img Next"))
        self.pushButton_14.setText(_translate("Form", "Res Next"))
        self.pushButton_15.setText(_translate("Form", "Res Prev"))
        self.pushButton_10.setText(_translate("Form", "-"))
        self.pushButton_2.setText(_translate("Form", "Clear"))
        self.pushButton.setText(_translate("Form", "Open Image"))
        self.pushButton_8.setText(_translate("Form", "Undo"))
        self.pushButton_7.setText(_translate("Form", "Open Mask"))
        
        # for segmentation
        self.label_5.setText(_translate("Form", "Class:"))
        self.pushButton_16.setText(_translate("Form", "+"))
        self.pushButton_17.setText(_translate("Form", "-"))
        self.pushButton_18.setText(_translate("Form", "Undo"))
        self.pushButton_19.setText(_translate("Form", "Clear"))
        self.pushButton_26.setText(_translate("Form", "Open Seg Map"))
        
        # for guidance selection
        self.label_6.setText(_translate("Form", "Sketch Map"))
        self.label_7.setText(_translate("Form", "Guidance:"))
        
        # for sketch
        self.checkBox_2.setText(_translate("Form", "Sketch"))
        self.checkBox.setText(_translate("Form", "Segmentation"))
        self.pushButton_20.setText(_translate("Form", "Undo"))
        self.pushButton_21.setText(_translate("Form", "Clear"))
        self.pushButton_22.setText(_translate("Form", "+"))
        self.pushButton_23.setText(_translate("Form", "-"))
        self.pushButton_24.setText(_translate("Form", "Erase"))
        self.pushButton_25.setText(_translate("Form", "Open Sketch Map"))
        