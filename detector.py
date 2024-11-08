from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np 

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        
        predictions = self.predictor(image)
                
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        panoptic_mask = panoptic_seg.to("cpu").numpy()
        
        print("Panoptic Segmentation Mask:", panoptic_mask)
        print("Segment Info:", segments_info)

        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))        
        output = viz.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        
        cv2.imshow("Panoptic Segmentation", output.get_image())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        