import json
import pickle
import cv2

IDX = 4

#folderpath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco/refcoco+/'
#images_folderpath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refer_data/images/mscoco/images/train2014/'

#folderpath = './RefCOCO_3DS/'
folderpath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/dataset_final/test/'
#images_folderpath = './RefCOCO_3DS/images/'
images_folderpath = folderpath + 'images/'

#filepath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/open_coco_data/annotations2017/instances_train2017.json'
#filepath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refer_data/refcoco/instances.json'

filepath = folderpath + 'instances.json'
with open(filepath, 'r') as fp:
    data = json.load(fp)

# Get the image
annotation = data['annotations'][IDX]
bbox = annotation['bbox']
image_id = int(annotation['image_id'])
image = data['images'][image_id]


# Reformat bbox
[x1, y1, x2, y2] = [int(i) for i in annotation['bbox']]

# Load the image from a PNG file
image_name = images_folderpath + '000006.png'
image = cv2.imread(image_name)
print([x1, y1, x2, y2])
print(image.shape)

width, height, _ = image.shape
#[x1, y1, x2, y2] = [width - x1, height - y1, width - x2, height - y2]
x2 = x1 + x2
y2 = y1 + y2
print([x1, y1, x2, y2])


# Define the pixel coordinates for the rectangle
#x1, y1 = 100, 100  # Top-left corner coordinates
#x2, y2 = 300, 300  # Bottom-right corner coordinates

# Draw the rectangle on the image
color = (0, 255, 0)  # Green color (BGR format)
thickness = 2  # Line thickness
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Display the image with the rectangle
output_filename = 'image_with_rectangle.png'
cv2.imwrite(output_filename, image)
#cv2.imshow('Image with Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# refs_data