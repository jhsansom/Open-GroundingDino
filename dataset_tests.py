import json
import pickle

#folderpath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco/refcoco+/'
#images_folderpath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refer_data/images/mscoco/images/train2014/'

folderpath = './RefCOCO_3DS/'
images_folderpath = './RefCOCO_3DS/images/'

#filepath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/open_coco_data/annotations2017/instances_train2017.json'
#filepath = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refer_data/refcoco/instances.json'

filepath = folderpath + 'instances.json'
with open(filepath, 'r') as fp:
    data = json.load(fp)

# keys = ['info', 'images', 'licenses', 'annotations', 'categories']
base = 0
#for i in range(base, base+10):
#    print()
#    print(data['images'][i])
#    print(data['annotations'][i])

#print()
#print(data['licenses'])
#print(data['info'])
#print(data['categories'])

#print()
#print(data.keys())

new_data = {
    'info' : data['info'],
    'images' : data['images'][:10],
    'annotations' : data['annotations'][:10],
    'categories' : data['categories']
}

file = 'refs.json'
full_file = folderpath + file

with open(full_file, 'rb') as file:
    refs_data = json.load(file)

#data = data[:10]

#with open('test_instances.json', 'w') as fp:
#    json.dump(new_data, fp, indent=4)




import cv2

instances_data = data
#for i, item in enumerate(instances_data['annotations']):
#    if item['image_id'] == 98304:
#        print('='*50)
#        print(f'Item #{i}')
#        print(json.dumps(item, indent = 4))
image_name = instances_data['annotations'][2]
print(json.dumps(image_name, indent = 4))
print()
image_id = image_name['image_id']
ann_id = image_name['id']

print('#'*50)
print('#'*50)
print('#'*50)

i = 0
for j, item in enumerate(refs_data):
    if (item['image_id'] == image_id) and (item['ann_id'] == ann_id):
        print('='*50)
        print(f'Item #{j}')
        print(json.dumps(item, indent = 4))
        print(i)
        i += 1
        break

[x1, y1, x2, y2] = [int(i) for i in image_name['bbox']]

# Load the image from a PNG file
image_name = images_folderpath + '000000.png'
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