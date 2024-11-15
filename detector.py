from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Detector:
    def __init__(self):
        """
        Creates Detector Object that is used for object segmentation
        """
        
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
    
    def classifyFrames(self, folderPath):
        """
        Classifies each frame in a given directory and outputs a segmented image
        
        Parameters:
        - folderPath: Directory where frames are located
        
        """
        def createCmap(self, segments_info):
            """

            Args:
                segments_info: Information on all of the segments in the picture, given by the predictor

            Returns:
                ListedColorMap: cmap that can be used by matplotlib
            """
            
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])            
            colors = [[0,0,0]]
            for object in segments_info:
                idx = object.get('category_id')
                if object.get('isthing') :
                    colors = colors + [metadata.get("thing_colors")[idx]]
                else:
                    colors = colors + [metadata.get("stuff_colors")[idx]]
            colors = np.array(colors)
            return ListedColormap(colors/255, name="custom") # Creates Custom cmap for plt based of metadatacatalog colours.
        
        def cropImage(image):
            """
            Parameters:
            - image: Input image with white border to be cropped

            Returns:
            - image: Cropped image
            """
            temp = np.where(image != [255,255,255]) 
            x1, x2, y1, y2 = temp[1].min(), temp[1].max(), temp[0].min(), temp[0].max()
            image = image[y1:y2,x1:x2] #Removes White Border that MatPlotLib puts on images
            return image
        
        for frame in os.listdir(folderPath):
            image = cv2.imread(os.path.join(folderPath,frame)) # Reads Frames from Folder
            
            predictions = self.predictor(image)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            panoptic_mask = panoptic_seg.to("cpu").numpy()#Produces Segmented Mask From Predictor
            
            cmap = createCmap(self, segments_info)
            
            fig, ax = plt.subplots(dpi=1200)     
            ax.imshow(panoptic_mask, cmap)
            ax.grid(False)
            ax.axis('off') #Produce Segmented Image
            
            fig.canvas.draw()
            panoptic_segmented_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            panoptic_segmented_image = panoptic_segmented_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            panoptic_segmented_image = cv2.cvtColor(panoptic_segmented_image, cv2.COLOR_BGR2RGB)
            plt.close(fig)#Image to array
            
            panoptic_segmented_image = cropImage(panoptic_segmented_image)
            
            print(f'Image {frame} segmented')
            cv2.imwrite('results/'+frame.split('.')[0]+'SEG.jpg', panoptic_segmented_image)#Saves images, path can be changed
       