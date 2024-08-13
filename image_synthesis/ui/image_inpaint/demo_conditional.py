import os
import sys
from click import edit
import cv2
import time
import numpy as np
from PIL import Image
import glob
from numpy.lib.function_base import diff
import torch
from torchvision.utils import save_image
import copy


# from image_synthesis.ui.image_inpaint.ui_util.ui_seg import Ui_Form
from image_synthesis.ui.image_inpaint.ui_util.ui_conditional import Ui_Form
from image_synthesis.ui.image_inpaint.ui_util.mouse_event import GraphicsScene

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

color_list = [QColor(255, 255, 255), QColor(0, 0, 0)]

def Color2Class(seg_mask, metadata):
    class_seg = np.zeros((seg_mask.shape[0], seg_mask.shape[1]))
    seg_mask = seg_mask.astype(np.int32)

    seg_mask = seg_mask[..., 0] + seg_mask[..., 1] * 256 + seg_mask[..., 2] * 256 * 256
    
    for i, color in enumerate(metadata.stuff_colors):
        c = color[0] + color[1] * 256 + color[2] * 256 * 256
        class_seg += (seg_mask == c) * i

    return class_seg.astype(np.uint8)


class Visualizer_draw(Visualizer):
    def draw_sem_seg(self, sem_seg):
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()

        # result = np.zeros_like(sem_seg)
        result = np.zeros((*sem_seg.shape, 3))
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None
            mask_color = np.array(mask_color)
            binary_mask = (sem_seg == label).astype(np.uint8)
            result += binary_mask[:, :, np.newaxis] * mask_color.reshape(1,1,3)

        return (result*255).astype(np.uint8)

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
        self.sketch_size = 1
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
        
        self.num_text_classes = 10

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

        self.seg_scene = GraphicsScene(1, self.size)
        self.graphicsView_4.setScene(self.seg_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.seg_scene_class = QGraphicsScene()
        self.graphicsView_5.setScene(self.seg_scene_class)
        self.graphicsView_5.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_5.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_5.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.sketch_scene = GraphicsScene(1, self.sketch_size)
        self.graphicsView_6.setScene(self.sketch_scene)
        self.graphicsView_6.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_6.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_6.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

        # init others
        self.img_path = '/home/liuqk/Program/python/image-synthesis/RESULT'
        self.mask_path = '/home/liuqk/Program/python/image-synthesis/RESULT'

        panoptic_metadata = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg")

        for i in range(self.num_unknown_classes):
            panoptic_metadata.stuff_classes.append('unknown'+str(i+1))
            # c = int(i * 10)
            # c = 
            # c = int((20-i-1) * 10)
            c = (20- i) * 829068
            # import pdb; pdb.set_trace()
            panoptic_metadata.stuff_colors.append([c // 255 // 255, c // 255 % 255, c % 255])

        self.metadata = panoptic_metadata
        self.stuff_classes = self.metadata.stuff_classes
        self.comboBox.addItems(sorted(self.metadata.stuff_classes))
        self.image_next()

    def open(self, im_path=None):
        # print('clicl open! {}'.format(im_path))
        if isinstance(im_path, str) and os.path.isfile(im_path):
            fileName = im_path
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.img_path)

        self.img_path = fileName
        if self.data_paths is not None:
            if self.img_path in self.data_paths['im_paths']:
                idx = self.data_paths['im_paths'].index(self.img_path)
                self.img_index = idx

        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName).convert('RGB').resize((256,256), resample=Image.BILINEAR)

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

            self.label.setText('Image {}'.format(os.path.basename(fileName)))

    def open_mask(self, mask_path=None):
        # print('clicl open! {}'.format(mask_path))
        if isinstance(mask_path, str) and os.path.isfile(mask_path):
            fileName = mask_path
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.mask_path)

        self.mask_path = fileName
        # if self.data_paths is not None:
        #     if self.img_path in self.data_paths['im_paths']:
        #         idx = self.data_paths['im_paths'].index(self.img_path)
        #         self.img_index = idx

        if fileName:
            mat_img = Image.open(fileName).convert('L').resize((256,256), resample=Image.NEAREST)
            mat_img = np.array(mat_img) # [H, W, 3]
            self.sketch = mat_img.copy()
            self.sketch_m = mat_img.copy()
            # process the new image
            img_np = np.array(self.img)
            mat_img[mat_img==255] = 1
            new_img_np = img_np * mat_img[:, :, np.newaxis]
            self.new_img_np = new_img_np

            self.show_seg_result(img_np, mat_img)
            self.show_sketch_result(img_np, mat_img)

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

    def open_seg_map(self, seg_map_path=None):
        # print('clicl open! {}'.format(mask_path))
        if isinstance(seg_map_path, str) and os.path.isfile(seg_map_path):
            fileName = seg_map_path
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.mask_path)

        if fileName:
            seg = Image.open(fileName).resize((256,256), resample=Image.NEAREST)
            seg = np.array(seg) # [H, W, 3]
            self.show_seg_result(seg_map=seg)


    def open_sketch_map(self, sketch_map_path=None):
        # print('clicl open! {}'.format(mask_path))
        if isinstance(sketch_map_path, str) and os.path.isfile(sketch_map_path):
            fileName = seg_map_path
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.mask_path)

        if fileName:
            sketch = Image.open(fileName).convert('L').resize((256,256), resample=Image.NEAREST)
            sketch = np.array(sketch) # [H, W]
            self.show_sketch_result(sketch_map=sketch)


    def bg_mode(self):
        self.sk_scene.mode = 0

    def sketch_mode(self):
        self.sk_scene.mode = 1

    def increase(self):
        if self.sk_scene.size < 25:
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

            self.sketch_map_array = self.make_mask(self.sketch_map_array, self.sketch_scene.mask_points[1], self.sketch_scene.size_points[1], 0)
            self.sketch_map_array = self.make_mask(self.sketch_map_array, self.sketch_scene.mask_points[0], self.sketch_scene.size_points[0], 255)
        
            mask = Image.fromarray(np.uint8(self.sketch_m))
            self.mask = mask

            im = self.img

            seg = self.seg_results[2]

            seg_mode = self.checkBox.isChecked()
            sketch_mode = self.checkBox_2.isChecked()

            if seg_mode and sketch_mode:
                mode = 3
            elif seg_mode:
                mode = 1
            elif sketch_mode:
                mode = 2
            else:
                mode = 0

            if self.model is not None:
                data = self.prepare_im_and_mask_func(**{'args': self.args, 'im': im, 'mask': mask, 'seg_result': seg, 'sketch': self.sketch_map_array})
                results = self.inpaint_func(**{'args': self.args, 'model': self.model, 'data': data, 'mode': mode})['completed']
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

    def edit_sketch(self):
        edit_sketch_mask = np.ones((256, 256))
        edit_sketch_mask = self.make_mask(edit_seg_mask, self.seg_scene.mask_points[1], self.seg_scene.size_points[1], 0)
        edit_sketch_mask = self.make_mask(edit_seg_mask, self.seg_scene.mask_points[0], self.seg_scene.size_points[0], 255)

        self.sketch_map[edit_sketch_mask == 0] = 0

        image3 = QImage(self.sketch_map, 256, 256, QImage.Format_Grayscale8)

        if image3.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image3)  
        sketch_map = pixmap.scaled(self.graphicsView_6.size(), Qt.IgnoreAspectRatio)
        if len(self.seg_scene_class.items())>0:
            self.seg_scene_class.removeItem(self.seg_scene_class.items()[-1])
            # self.seg_scene_class.reset_items()
        self.sk_scene.addPixmap(self.sk_image) 
        self.sketch_scene.addPixmap(sketch_map)


    def edit_seg(self):
        _, _, class_seg = self.seg_results

        edit_seg_mask = np.ones((256, 256))
        edit_seg_mask = self.make_mask(edit_seg_mask, self.seg_scene.mask_points[1], self.seg_scene.size_points[1], 0)
        edit_seg_mask = self.make_mask(edit_seg_mask, self.seg_scene.mask_points[0], self.seg_scene.size_points[0], 255)
        
        choice = self.comboBox.currentText()

        idx = self.stuff_classes.index(choice)
        class_seg = class_seg * edit_seg_mask + (1 - edit_seg_mask) * idx
        class_seg = class_seg.astype(self.seg_results[-1].dtype)

        visualizer = Visualizer_draw(self.new_img_np, self.metadata)
        visualizer_text = Visualizer(self.img, self.metadata)
        vis_output = visualizer_text.draw_sem_seg(class_seg)
        show_img = vis_output.get_image()
        show_img_class = visualizer.draw_sem_seg(class_seg)
        self.seg_results = (show_img, show_img_class, class_seg)

        image2 = QImage(show_img, 256, 256, QImage.Format_RGB888)

        if image2.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image2)  
        self.seg_image = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
        if len(self.seg_scene.items())>0:
            self.seg_scene.removeItem(self.seg_scene.items()[-1])
        # self.sk_scene.addPixmap(self.sk_image) 
        self.seg_scene.addPixmap(self.seg_image)

        image3 = QImage(show_img_class, 256, 256, QImage.Format_RGB888)

        if image3.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image3)  
        self.seg_image_class = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
        if len(self.seg_scene_class.items())>0:
            self.seg_scene_class.removeItem(self.seg_scene_class.items()[-1])
        # self.sk_scene.addPixmap(self.sk_image) 
        self.seg_scene_class.addPixmap(self.seg_image_class)
        self.seg_scene.reset()

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

        if np.array(self.mask).ndim != 3:
            masked_img = np.array(self.mask)[:,:,np.newaxis] / 255 * np.array(self.img)
        else:
            masked_img = np.array(self.mask) / 255 * np.array(self.img)
        Image.fromarray(masked_img.astype(np.uint8)).save(os.path.join(save_dir, 'masked_image.png'))
        visualizer = Visualizer_draw(masked_img, self.metadata)
        visualizer_text = Visualizer(masked_img, self.metadata)
        # panoptic_seg, segments_info, class_seg = self.seg_results
        _, _, class_seg = self.seg_results
        # vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), segments_info)
        seg_map = visualizer.draw_sem_seg(class_seg)
        seg_map_text = visualizer_text.draw_sem_seg(class_seg)
        # seg_map = vis_output.get_image()
        sketch_map = self.sketch_map_array

        Image.fromarray(seg_map.astype(np.uint8)).save(os.path.join(save_dir, '{}_seg_map.png'.format(mask_count)))
        Image.fromarray(seg_map_text.get_image().astype(np.uint8)).save(os.path.join(save_dir, '{}_seg_map_text.png'.format(mask_count)))
        Image.fromarray(sketch_map.astype(np.uint8)).save(os.path.join(save_dir, '{}_edge_map.png'.format(mask_count)))

        self.mask.save(os.path.join(save_dir, '{}_mask.png'.format(mask_count)))
        print('saved to {}'.format(os.path.join(save_dir, '{}_mask.png'.format(mask_count))))

        if len(self.result_img) > 0:
            seg_mode = self.checkBox.isChecked()
            sketch_mode = self.checkBox_2.isChecked()
            if seg_mode and sketch_mode:
                condition = 'seg_ske'
            elif seg_mode:
                condition = 'seg'
            elif sketch_mode:
                condition = 'ske'
            else:
                condition = 'none'
            for i in range(len(self.result_img)):
                save_path = os.path.join(save_dir, '{}_completed_{}_{}.png'.format(mask_count, str(i).zfill(2), condition))
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
            self.label_3.setText('Result {}/{}'.format(self.result_index+1, len(self.result_img)))
    
    def result_next(self):
        if self.result_img is not None and len(self.result_img) > 0:
            self.result_index += 1
            self.result_index = self.result_index % len(self.result_img)
            pixel_map = self.result_img[self.result_index].toqpixmap()
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(pixel_map)
            self.label_3.setText('Result {}/{}'.format(self.result_index+1, len(self.result_img)))


    def image_previous(self):
        if self.data_paths is not None and len(self.data_paths['im_paths']) > 0:
            self.img_index -= 1
            self.img_index = self.img_index % len(self.data_paths['im_paths'])
            im_path = self.data_paths['im_paths'][self.img_index]
            self.open(im_path)
        if self.data_paths is not None and len(self.data_paths['mask_paths']) > 0:
            mask_path = self.data_paths['mask_paths'][self.img_index]
            self.open_mask(mask_path)
        self.label.setText('Image {}/{}, {}'.format(self.img_index+1, len(self.data_paths['im_paths']), os.path.basename(im_path)))


    def image_next(self):
        if self.data_paths is not None and len(self.data_paths['im_paths']) > 0:
            self.img_index += 1
            self.img_index = self.img_index % len(self.data_paths['im_paths'])
            im_path = self.data_paths['im_paths'][self.img_index]
            self.open(im_path)
        if self.data_paths is not None and len(self.data_paths['mask_paths']) > 0:
            mask_path = self.data_paths['mask_paths'][self.img_index]
            self.open_mask(mask_path)
        self.label.setText('Image {}/{}, {}'.format(self.img_index+1, len(self.data_paths['im_paths']), os.path.basename(im_path)))

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

    def show_sketch_result(self, img=None, mask=None, sketch_map=None):
        # if self.model.sketch_model is None:
        #     self.model.init_sketch_model()
        if sketch_map is None:
            self.sketch_scene.reset()
            img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).unsqueeze(0).to(self.model.device)
            # sketch_map = self.model.sketch_model(img[:, [2,1,0], :, :])
            # sketch_map = sketch_map.cpu().numpy()[0,0,:,:]
            sketch_map = self.model.sketch_model(img[:,[2,1,0],:,:])
            # sketch_map = (1-sketch_map) * 255
            sketch_map = sketch_map.cpu().numpy()[0,0,:,:]

        # sketch_map = sketch_map * mask + 255 * (1-mask)
        # sketch_map = sketch_map 
        sketch_map = sketch_map.astype(np.uint8)
        self.sketch_map_array = sketch_map
        # self.sketch_map = sketch_map

        image3 = QImage(sketch_map, 256, 256, QImage.Format_Grayscale8)

        if image3.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image3)  
        sketch_map = pixmap.scaled(self.graphicsView_6.size(), Qt.IgnoreAspectRatio)
        self.sketch_map = sketch_map
        # if len(self.seg_scene_class.items())>0:
        #     # self.seg_scene_class.removeItem(self.seg_scene_class.items()[-1])
        #     self.seg_scene_class.reset_items()
        # self.sk_scene.addPixmap(self.sk_image) 
        self.sketch_scene.addPixmap(sketch_map)
        

    def show_seg_result(self, img=None, mask=None, seg_map=None):
        # img = np.array(img)[:, :, ::-1]
        # img = torch.tensor(img).unsqueeze(0).to(self.model.device)
        # mask = torch.tensor(mask).unsqueeze(0).to(self.model.device)

        if seg_map is None:
            original_img = copy.deepcopy(img)
            
            img = img * mask[:, :, np.newaxis]
            visualizer = Visualizer_draw(img, self.metadata)
            visualizer_text = Visualizer(original_img, self.metadata)
            # visualizer = Visualizer(original_img, self.metadata)

            img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).unsqueeze(0)
            # height, width = img.shape[:2]
            mask = torch.as_tensor(mask.astype("float32"))[:,:].unsqueeze(0).unsqueeze(0)

            original_img = torch.as_tensor(original_img.astype("float32").transpose(2,0,1)).unsqueeze(0)
            # inputs = {"image": img, "height": height, "width": width}
            # panoptic_seg, segments_info = self.model.guid_model(img, mask)[0]['panoptic_seg']
            # import pdb; pdb.set_trace()
            class_seg = self.model.seg_model(original_img, None)
            class_seg = class_seg.cpu()[0].numpy()
        else:
            visualizer = Visualizer_draw(self.new_img_np, self.metadata)
            visualizer_text = Visualizer(self.img, self.metadata)
            #TODO:
            class_seg = Color2Class(seg_map, self.metadata)
        
        # self.seg_results = class_seg
        show_img_class = visualizer.draw_sem_seg(class_seg)
        vis_output = visualizer_text.draw_sem_seg(class_seg)
        show_img = vis_output.get_image()
        self.seg_results = (show_img, show_img_class, class_seg)

        image2 = QImage(show_img, 256, 256, QImage.Format_RGB888)

        if image2.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image2)  
        self.seg_image = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
        if len(self.seg_scene.items())>0:
            self.seg_scene.removeItem(self.seg_scene.items()[-1])
            # self.seg_scene.reset()
        # self.sk_scene.addPixmap(self.sk_image) 
        self.seg_scene.addPixmap(self.seg_image)

        image3 = QImage(show_img_class, 256, 256, QImage.Format_RGB888)

        if image3.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image3)  
        self.seg_image_class = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
        if len(self.seg_scene_class.items())>0:
            self.seg_scene_class.removeItem(self.seg_scene_class.items()[-1])
            # self.seg_scene_class.reset_items()
        self.sk_scene.addPixmap(self.sk_image) 
        self.seg_scene_class.addPixmap(self.seg_image_class)


    def increase_seg(self):
        if self.seg_scene.size < 25:
            self.seg_scene.size += 1
    
    def decrease_seg(self):
        if self.seg_scene.size > 1:
            self.seg_scene.size -= 1

    def undo_seg(self):
        self.seg_scene.undo()

    def clear_seg(self):
        # self.sketch_m = self.sketch.copy()
        # mat_img = (np.zeros((256,256,3)) + 255).astype(np.uint8)
        # self.sketch = mat_img.copy()
        # self.sketch_m = mat_img.copy()
    
        self.seg_scene.reset_items()
        self.seg_scene.reset()
        # self.mask = None
        # if type(self.sk_image):
        #     self.sk_scene.addPixmap(self.sk_image)
        if type(self.seg_image_class):
            self.seg_scene.addPixmap(self.seg_image)

    def undo_sketch(self):
        self.sketch_scene.undo()
    
    def clear_sketch(self):
        self.sketch_scene.reset_items()
        self.sketch_scene.reset()
        # self.mask = None
        if type(self.sketch_map):
            self.sketch_scene.addPixmap(self.sketch_map)
        # self.show_sketch_result()

    def increase_sketch(self):
        if self.sketch_scene.size < 25:
            self.sketch_scene.size += 1

    def decrease_sketch(self):
        if self.sketch_scene.size > 1:
            self.sketch_scene.size -= 1

    def erase_sketch(self):
        sketch_map = self.sketch_map_array

        edit_sketch_mask = np.ones((256, 256))
        edit_sketch_mask = self.make_mask(edit_sketch_mask, self.sketch_scene.mask_points[1], self.sketch_scene.size_points[1], 0)
        # edit_sketch_mask = self.make_mask(edit_sketch_mask, self.sketch_scene.mask_points[0], self.sketch_scene.size_points[0], 255)
        
        sketch_map = edit_sketch_mask * sketch_map + 255 * (1 - edit_sketch_mask)

        self.sketch_map_array = sketch_map.astype(np.uint8)

        image3 = QImage(self.sketch_map_array, 256, 256, QImage.Format_Grayscale8)

        if image3.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot show seg results")
            return    
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image3)  
        sketch_map = pixmap.scaled(self.graphicsView_6.size(), Qt.IgnoreAspectRatio)
        self.sketch_map = sketch_map
        # if len(self.seg_scene_class.items())>0:
        #     # self.seg_scene_class.removeItem(self.seg_scene_class.items()[-1])
        #     self.seg_scene_class.reset_items()
        # self.sk_scene.addPixmap(self.sk_image) 
        self.sketch_scene.addPixmap(sketch_map)
        self.sketch_scene.reset()



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    app = QApplication(sys.argv)
    ex = Ex()
    sys.exit(app.exec_())

