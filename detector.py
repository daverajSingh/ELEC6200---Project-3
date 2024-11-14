from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
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

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, None, fx = 0.1, fy = 0.1)
        imageName = imagePath.split('/')[-1].split('.')[0]
        
        predictions = self.predictor(image)
                
        panoptic_seg, segments_info = predictions["panoptic_seg"]
                
        panoptic_mask = panoptic_seg.to("cpu").numpy()
        
        print(panoptic_mask, segments_info)
        
        unique_values = np.unique(panoptic_mask)
        colors = ['black'] + list(plt.cm.tab20.colors[:len(unique_values) - 1])
        cmap = ListedColormap(colors)
                
        plt.imshow(panoptic_mask, cmap)
        plt.grid(False)
        plt.axis('off')
        plt.savefig(imageName+'ClassifiedPLT.jpg', dpi=1000)

        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))        
        output = viz.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info)
        
        cv2.imshow("segment", output.get_image())
        cv2.waitKey(0)
        cv2.imwrite(imageName+'Classified.jpg', output.get_image())
        cv2.destroyAllWindows()
        
    
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
