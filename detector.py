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
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        
        self.cfg.TEST.AUG.ENABLED = True
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
    
    def classifyFrames(self, folderPath, outputPath):
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
        
        os.makedirs(outputPath, exist_ok=True)
        for filename in os.listdir(outputPath):
            file_path = os.path.join(outputPath, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
        for frame in sorted([f for f in os.listdir(folderPath) if f.endswith('.jpg') or f.endswith('.png')]):
            image = cv2.imread(os.path.join(folderPath, frame))  # Reads Frames from Folder

            # Get predictions
            predictions = self.predictor(image)
            instances = predictions["instances"].to("cpu")
            
            masks = instances.pred_masks.numpy()
            classes = instances.pred_classes.numpy()

            filtered_classes = ["bottle", "mouse", "keyboard", "bowl", "book"]
            
            # Prepare composite mask
            segmented_image = np.zeros_like(image, dtype=np.uint8)
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            
            filtered_class_ids = [metadata.thing_classes.index(cls) for cls in filtered_classes]            
            for mask, class_id in zip(masks, classes):
                if class_id in filtered_class_ids:
                    color = metadata.thing_colors[class_id]
                    segmented_image[mask] = color  # Apply class color to mask region

            # Save the segmented image
            print(f'Image {frame} segmented')
            cv2.imwrite(os.path.join(outputPath, frame.split('.')[0] + 'SEG.jpg'), segmented_image)
