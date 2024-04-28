import pickle
import json
import tqdm
import numpy as np
import os

spatial_terms = [
    'right',
    'rightmost',
    'left',
    'leftmost',
    'furthest',
    'farthest',
    'far',
    'near',
    'nearest',
    'top',
    'topmost',
    'upper',
    'uppermost',
    'bottom',
    'close',
    'closest',
    'lower',
    'foreground',
    'background',
    'distant'
]


class DatasetSplitter:

    def __init__(self, instances_path, refs_path, out_folder):
        # Constants
        self.instances_path = instances_path
        self.refs_path = refs_path
        self.out_folder = out_folder

        self.load_data()

        self.print_dataset_statistics()

    # Load the source data
    def load_data(self):
        with open(self.instances_path, 'r') as fp:
            self.instances_data = json.load(fp)
        
        if self.refs_path[-1] == 'p': # this means its pickled
            with open(self.refs_path, 'rb') as fp:
                self.refs_data = pickle.load(fp)
        else: # this means its a regular json
            with open(self.refs_path, 'r') as fp:
                self.refs_data = json.load(fp)

    def save_data(self):
        instances_path = self.out_folder + 'instances.json'
        refs_path = self.out_folder + 'refs.json'

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        with open(instances_path, 'w') as fp:
            json.dump(self.instances_data, fp)
        with open(refs_path, 'w') as fp:
            json.dump(self.refs_data, fp)

    # Get train, val, test split 
    def get_tvt_split(self, split):
        refs = []
        for ref in self.refs_data:
            if ref['split'] == split:
                refs.append(ref)
        self.refs_data = refs

    # Load a randomized subset of the data
    def get_random_subset(self, num_images):
        num_current_images = len(self.instances_data['images'])
        assert num_images < num_current_images

        random_subset = np.random.randint(0, high=num_current_images, size=num_images)

        self.instances_data['images'] = [self.instances_data['images'][idx] for idx in random_subset]

        valid_image_ids = [img['id'] for img in self.instances_data['images']]

        anns = []
        print('Removing annotations that dont correspond to valid image')
        for ann in tqdm.tqdm(self.instances_data['annotations']):
            if ann['image_id'] in valid_image_ids:
                anns.append(ann)

        self.instances_data['annotations'] = anns

        valid_ann_ids = [a['id'] for a in self.instances_data['annotations']]

        refs = []
        print('Removing references that dont correspond to valid annotation')
        for ref in tqdm.tqdm(self.refs_data):
            if ref['ann_id'] in valid_ann_ids:
                refs.append(ref)

        self.refs_data = refs

    def get_image_with_anns(self):
        valid_image_ids = [a['image_id'] for a in self.instances_data['annotations']]

        images = []
        print('Getting valid images')
        for image in tqdm.tqdm(self.instances_data['images']):
            if image['id'] in valid_image_ids:
                images.append(image)
    
        self.instances_data['images'] = images


    # Restrict annotations to those with corresponding references
    def get_ann_w_refs(self):
        ann_ids = [ref['ann_id'] for ref in self.refs_data]
        new_annotations = []
        print('Reducing annotations')
        for a in tqdm.tqdm(self.instances_data['annotations']):
            if a['id'] in ann_ids:
                new_annotations.append(a)
        print('Finished reducing annotations')
        self.instances_data['annotations'] = new_annotations

    # Split spatial vs. non-spatial
    def split_spatial_nonspatial(self, spatial=True):
        refs = []
        for ref in self.refs_data:
            sentences = []
            for sentence in ref['sentences']:
                intersection = list(set(sentence['tokens']) & set(spatial_terms))
                if (len(intersection) > 0) == spatial:
                    sentences.append(sentence)
            if len(sentences) > 0:
                ref['sentences'] = sentences
                refs.append(ref)
        self.refs_data = refs

    # Print out dataset statistics
    def print_dataset_statistics(self):
        num_images = len(self.instances_data['images'])
        print(f'Number of images = {num_images}')
        num_annotations = len(self.instances_data['annotations'])
        print(f'Number of annotations = {num_annotations}')
        num_refs = len(self.refs_data)
        print(f'Number of refs = {num_refs}')

if __name__ == '__main__':
    #INSTANCES_PATH = "/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco/refcoco/instances.json"
    #REFS_PATH = "/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco/refcoco/refs(google).p"
    #FOLDER_PATH = "/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/synthetic_data/split_datasets/RefCOCO_3ds_7k/train/"
    FOLDER_PATH = "/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco_split/valid_annotations/val/"
    INSTANCES_PATH = FOLDER_PATH + "instances.json"
    REFS_PATH = FOLDER_PATH + "refs.json"
    OUT_FOLDER = "/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco_split/nonspatial" + '/' # <--- include this slash

    ds = DatasetSplitter(INSTANCES_PATH, REFS_PATH, OUT_FOLDER)
    #ds.get_tvt_split('val')
    ds.split_spatial_nonspatial(spatial=False)
    ds.get_ann_w_refs()
    #ds.print_dataset_statistics()
    #ds.get_random_subset(5000)
    ds.get_image_with_anns()
    ds.print_dataset_statistics()
    ds.save_data()