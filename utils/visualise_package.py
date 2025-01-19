import numpy as np
import cv2
from PIL import Image

from extract_labels import process_image_with_global_mapping, create_global_label_mapping


data_path = 'nerf_formated_data_small.npz'


mapper_seg = {(0, 0, 0): 0, (100, 56, 103): 1, (134, 56, 103): 2}
mapper_seg = create_global_label_mapping("./bottlenmouse/images_segmented")
print(mapper_seg)
seg_input = Image.open("./bottlenmouse/images_segmented/00011SEG.jpg")

new_seg = np.array(process_image_with_global_mapping(seg_input, mapper_seg), dtype=np.float32)


loaded = np.load(data_path)
images = loaded['images_train'] 
# seg_images = loaded['seg_images']
# new_seg = new_seg == 1
print(np.unique(new_seg))

new_seg = (new_seg * (255 // 4)).astype(np.uint8)
cv2.imshow('image', images[0])

print(np.unique(new_seg))
cv2.imshow('seg', new_seg)

cv2.waitKey(0)