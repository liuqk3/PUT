import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
import glob
from numpy.lib.function_base import diff
import torch
from torchvision.utils import save_image


from image_synthesis.ui.image_inpaint.ui_util.ui import Ui_Form
from image_synthesis.ui.image_inpaint.ui_util.mouse_event import GraphicsScene

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

color_list = [QColor(255, 255, 255), QColor(0, 0, 0)]

class Ex(QWidget, Ui_Form):
    def __init__(
        self, 
        args=None, 
        model=None,
        data_paths=None,
        prepare_im_and_mask_func=None,
        inpaint_func=None,

        ):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.args = args
        self.data_paths = data_paths
        self.prepare_im_and_mask_func = prepare_im_and_mask_func
        self.inpaint_func = inpaint_func

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.sketch = None
        self.sketch_m = None
        self.img = None
        self.img_index = -1
        self.img_path = None
        self.mask = None
        self.mask_path = None
        self.result_img = []
        self.result_index = -1
        


        # init the graphics
        self.mouse_clicked = False
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.sk_scene = GraphicsScene(1, self.size)
        self.graphicsView_2.setScene(self.sk_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

        # # init others
        # self.img_path = '/home/liuqk/Program/python/image-synthesis/RESULT'
        # self.mask_path = '/home/liuqk/Program/python/image-synthesis/RESULT'

        self.image_next()

    def open(self, im_path=None):
        # print('clicl open! {}'.format(im_path))
        if isinstance(im_path, str) and os.path.isfile(im_path):
            fileName = im_path
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.img_path)

        self.img_path = fileName
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName).resize((256,256), resample=Image.BILINEAR)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.scene.items())>0:
                self.scene.removeItem(self.scene.items()[-1])
            self.scene.addPixmap(image)
            
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image)
            
            # init mask
            mat_img = np.zeros((256,256,3)).astype(np.uint8)
            self.sketch = mat_img.copy()
            self.sketch_m = mat_img.copy()
            image2 = QImage(mat_img, 256, 256, QImage.Format_RGB888)

            if image2.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(256):
                for j in range(256): 
                    image2.setPixel(i, j, color_list[0].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image2)  
            # self.sk_image = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
            self.sk_image = image

            if len(self.sk_scene.items())>0:
                self.sk_scene.removeItem(self.sk_scene.items()[-1])
            self.sk_scene.addPixmap(self.sk_image) 

            self.label0.setText('Image {}'.format(os.path.basename(fileName)))

    def open_mask(self, mask_path=None):
        # print('clicl open! {}'.format(mask_path))
        if isinstance(mask_path, str) and os.path.isfile(mask_path):
            fileName = mask_path
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.mask_path)

        self.mask_path = fileName
        if fileName:
            mat_img = Image.open(fileName).convert('RGB').resize((256,256), resample=Image.NEAREST)
            mat_img = np.array(mat_img) # [H, W, 3]
            self.sketch = mat_img.copy()
            self.sketch_m = mat_img.copy()
            # process the new image
            img_np = np.array(self.img)
            mat_img[mat_img==255] = 1
            new_img_np = img_np * mat_img
            # show in self.sk_scene
            image2 = QImage(new_img_np, 256, 256, QImage.Format_RGB888)

            if image2.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image2)  
            self.sk_image = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
            if len(self.sk_scene.items())>0:
                self.sk_scene.removeItem(self.sk_scene.items()[-1])
            # self.sk_scene.addPixmap(self.sk_image) 
            self.sk_scene.addPixmap(self.sk_image)


    def bg_mode(self):
        self.sk_scene.mode = 0

    def sketch_mode(self):
        self.sk_scene.mode = 1

    def increase(self):
        if self.sk_scene.size < 15:
            self.sk_scene.size += 1
    
    def decrease(self):
        if self.sk_scene.size > 1:
            self.sk_scene.size -= 1

    def edit(self):
        # process mask
        # for i in [0, 1]:
        #     self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[i], self.sk_scene.size_points[i], i)
        if self.img is not None:
            print('start editing!')
            self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[1], self.sk_scene.size_points[1], 0)
            self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[0], self.sk_scene.size_points[0], 255)
        
            mask = Image.fromarray(np.uint8(self.sketch_m))
            self.mask = mask

            im = self.img

            if self.model is not None:
                data = self.prepare_im_and_mask_func(**{'args': self.args, 'im': im, 'mask': mask})
                results = self.inpaint_func(**{'args': self.args, 'model': self.model, 'data': data})['completed']
                results = results.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                results_ = []
                for idx in range(results.shape[0]):
                    res = Image.fromarray(results[idx])
                    results_.append(res)
                self.result_img = results_
                print('Inpainting done!')
                self.result_index = -1
                self.result_next()
            # mask.save('mask.png')

    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'], (color, color, color), sizes[idx])
        return mask

    def save_img(self):
        # if type(self.output_img):
        #     fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
        #             QDir.currentPath())
        #     cv2.imwrite(fileName+'.jpg',self.output_img
        if self.img is None:
            return 
        


        basename = os.path.basename(self.img_path).replace('.png', '').replace('.JPEG', '').replace('.jpg', '')
        save_dir = os.path.join(self.args.save_dir, basename)
        os.makedirs(save_dir, exist_ok=True)

        self.img.save(os.path.join(save_dir, basename+'.png'))
        
        mask_count = len(glob.glob(os.path.join(save_dir, '*_mask.png'))) + 1
        mask_count = str(mask_count).zfill(4)
        if self.mask is None:
            self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[1], self.sk_scene.size_points[1], 0)
            self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[0], self.sk_scene.size_points[0], 255)
        
            mask = Image.fromarray(np.uint8(self.sketch_m))
            self.mask = mask
        self.mask.save(os.path.join(save_dir, '{}_mask.png'.format(mask_count)))
        print('saved to {}'.format(os.path.join(save_dir, '{}_mask.png'.format(mask_count))))

        if len(self.result_img) > 0:
            for i in range(len(self.result_img)):
                save_path = os.path.join(save_dir, '{}_completed_{}.png'.format(mask_count, str(i).zfill(2)))
                self.result_img[i].save(save_path)
                print('saved to {}'.format(save_path))

    def result_previous(self):
        if self.result_img is not None and len(self.result_img) > 0:
            self.result_index -= 1
            if self.result_index < 0:
                self.result_index = self.result_index + len(self.result_img)
            pixel_map = self.result_img[self.result_index].toqpixmap()
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(pixel_map)
            self.label2.setText('Result {}/{}'.format(self.result_index+1, len(self.result_img)))
    
    def result_next(self):
        if self.result_img is not None and len(self.result_img) > 0:
            self.result_index += 1
            self.result_index = self.result_index % len(self.result_img)
            pixel_map = self.result_img[self.result_index].toqpixmap()
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(pixel_map)
            self.label2.setText('Result {}/{}'.format(self.result_index+1, len(self.result_img)))


    def image_previous(self):
        if self.data_paths is not None and len(self.data_paths['im_paths']) > 0:
            self.img_index -= 1
            self.img_index = self.img_index % len(self.data_paths['im_paths'])
            im_path = self.data_paths['im_paths'][self.img_index]
            self.open(im_path)
        if self.data_paths is not None and len(self.data_paths['mask_paths']) > 0:
            mask_path = self.data_paths['mask_paths'][self.img_index]
            self.open_mask(mask_path)
        self.label0.setText('Image {}/{}, {}'.format(self.img_index+1, len(self.data_paths['im_paths']), os.path.basename(im_path)))


    def image_next(self):
        if self.data_paths is not None and len(self.data_paths['im_paths']) > 0:
            self.img_index += 1
            self.img_index = self.img_index % len(self.data_paths['im_paths'])
            im_path = self.data_paths['im_paths'][self.img_index]
            self.open(im_path)
        if self.data_paths is not None and len(self.data_paths['mask_paths']) > 0:
            mask_path = self.data_paths['mask_paths'][self.img_index]
            self.open_mask(mask_path)
        self.label0.setText('Image {}/{}, {}'.format(self.img_index+1, len(self.data_paths['im_paths']), os.path.basename(im_path)))


    def undo(self):
        self.sk_scene.undo()

    def clear(self):
        # self.sketch_m = self.sketch.copy()
        mat_img = (np.zeros((256,256,3)) + 255).astype(np.uint8)
        self.sketch = mat_img.copy()
        self.sketch_m = mat_img.copy()
    
        self.sk_scene.reset_items()
        self.sk_scene.reset()
        self.mask = None
        # if type(self.sk_image):
        #     self.sk_scene.addPixmap(self.sk_image)
        if type(self.img):
            self.sk_scene.addPixmap(self.img.toqpixmap())

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    app = QApplication(sys.argv)
    ex = Ex()
    sys.exit(app.exec_())

