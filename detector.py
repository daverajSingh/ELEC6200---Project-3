from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)

    
    def classifyFrames(self, folderPath):
        for frame in os.listdir(folderPath):
            image = cv2.imread(os.path.join(folderPath,frame)) # Reads Frames from Folder
            
            predictions = self.predictor(image)
            panoptic_seg, _ = predictions["panoptic_seg"]
            panoptic_mask = panoptic_seg.to("cpu").numpy()#Produces Segmented Mask From Predictor
            
            unique_values = np.unique(panoptic_mask)
            colors = ['black'] + list(plt.cm.tab20.colors[:len(unique_values) - 1])
            cmap = ListedColormap(colors) # Colour Map for all objects in scene
               
            fig, ax = plt.subplots(dpi=1000)     
            ax.imshow(panoptic_mask, cmap)
            ax.grid(False)
            ax.axis('off') #Produce Segmented Image
            
            fig.canvas.draw()
            panoptic_segmented_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            panoptic_segmented_image = panoptic_segmented_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            panoptic_segmented_image = cv2.cvtColor(panoptic_segmented_image, cv2.COLOR_BGR2RGB)
            plt.close(fig)#Image to array
            
            temp = np.where(panoptic_segmented_image != [255,255,255]) 
            x1, x2, y1, y2 = temp[1].min(), temp[1].max(), temp[0].min(), temp[0].max()
            panoptic_segmented_image = panoptic_segmented_image[y1:y2,x1:x2] #Removes White Border that MatPlotLib puts on images
            
            cv2.imwrite(frame.split('.')[0]+'SEG.jpg', panoptic_segmented_image)#Saves images, path can be changed
