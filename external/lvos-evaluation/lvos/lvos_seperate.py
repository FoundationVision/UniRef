import json
import os
from collections import defaultdict
from glob import glob

import numpy as np
from PIL import Image


class LVOS(object):
    SUBSET_OPTIONS = ['valid', 'test']
    TASKS = ['semi-supervised',]
    DATASET_WEB = 'https://lingyihongfd.github.io/lvos.github.io/'
    VOID_LABEL = 255

    def __init__(self, root, task='semi-supervised', subset='val', codalab=False):
        """
        Class to read the LVOS dataset
        :param root: Path to the LVOS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages')
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.root, annotations_folder)

        self.img_path = self.root
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.root,annotations_folder)

        json_path=os.path.join(root,self.subset+'_meta.json')
            

        with open(json_path,'r') as f:
            self.json_data=json.load(f)
        self.json_data=self.json_data['videos']
        self.sequences_names = list(self.json_data.keys())
        sequences_names=sorted(self.sequences_names)
        
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            seq_data=self.json_data[seq]["objects"]

            self.sequences[seq]=seq_data
            objs=list(seq_data.keys())

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'LVOS not found in the specified directory, download it from {self.DATASET_WEB}')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found, download the missing subset '
                                    f'for the {self.task} task from {self.DATASET_WEB}')
        if self.subset in ['train', 'val'] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder for the {self.task} task not found, download it from {self.DATASET_WEB}')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq
    
    def get_sequence(self,sequence):
        tmp_sequence=dict()
        tmp_sequence[sequence]=self.sequences[sequence]
        return tmp_sequence

    def get_mask(self,sequence,frame, target_obj=None):
        masks =  np.array(Image.open(os.path.join(self.mask_path,sequence,frame+'.png')))
        masks=np.expand_dims(masks,axis=0)
        masks_void = np.zeros_like(masks)


        if target_obj is not None:
            tmp_masks=np.zeros_like(masks)
            tmp_masks[masks==int(target_obj)]=1
            masks=tmp_masks

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        return masks, masks_void





