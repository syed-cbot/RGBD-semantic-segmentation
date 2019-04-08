import os
#import yaml
#import json
from functools import reduce
import glob


class DAVIS():
    def __init__(self, data_dir, train_val='training'):
        self.data_dir = data_dir
        name_list = {
            'bike-packing', 'bmx-bumps', 'bmx-trees', 'breakdance-flare', 'cat-girl',
            'crossing', 'disc-jockey', 'dogs-jump', 'drone', 'hike', 'horsejump-high',
            'horsejump-low', 'kid-football', 'kite-surf', 'kite-walk', 'loading', 'longboard',
            'lab-coat', 'lucia', 'mbike-trick', 'miami-surf', 'motocross-jump', 'motorbike',
            'paragliding', 'paragliding-launch', 'parkour', 'rollerblade', 'schoolgirls',
            'shooting', 'snowboard', 'stroller', 'stunt', 'swing', 'tennis', 'tuk-tuk',
            'upside-down', 'walking',
        }
        if train_val=='training':
            y = [i.strip() for i in open(f'{data_dir}/ImageSets/2017/train.txt').readlines()]
        else:
            y = [i.strip() for i in open(f'{data_dir}/ImageSets/2017/val.txt').readlines()]
        self.image_list = []
        self.image = []
        self.label = []
        self.background_image = []
        self.background_label = []
        for sequence in y:
            if sequence not in name_list:
                continue
            for i in range(1, len(glob.glob(os.path.join(data_dir, 'JPEGImages/480p', sequence) + '/*.jpg')) - 1):
                self.image_list.append([
                    os.path.join(data_dir, 'JPEGImages/480p', sequence, f'{i-1:05d}.jpg'),
                    os.path.join(data_dir, 'JPEGImages/480p', sequence, f'{i:05d}.jpg'),
                    os.path.join(data_dir, 'JPEGImages/480p', sequence, f'{i+1:05d}.jpg'),
                ])
                self.image.append(
                    os.path.join(data_dir, 'JPEGImages/480p', sequence, f'{i:05d}.jpg')
                )
                self.label.append(
                    os.path.join(data_dir, 'anno_binary', sequence, f'{i:05d}.png')
                )
                self.background_image.append(
                    os.path.join(data_dir, 'JPEGImages/480p', sequence, f'{0:05d}.jpg')
                )
                self.background_label.append(
                    os.path.join(data_dir, 'anno_binary', sequence, f'{0:05d}.png')
                )

    def size(self):
        return len(self.image_list)


class YOUTUBE():
    def __init__(self, data_dir, train_val='training'):
        self.data_dir = data_dir
        self.image_list = []
        self.image = []
        self.label = []
        self.background_image = []
        self.background_label = []
        self.json_file = json.load(open(os.path.join(data_dir, 'train_zip/train/meta.json'),'r'))
        f = open(os.path.join(data_dir, 'list', train_val + '.txt'),'r').readlines()
        for video_name in f:
            video_name = video_name.split('\n')[0]
            keys = list(self.json_file['videos'][video_name]['objects'].keys())
            frames_list = self.json_file['videos'][video_name]['objects'][keys[0]]['frames']
            for i in range(1,len(frames_list)-1):
                base_dir = os.path.join(data_dir, 'train_all_frames_zip/train_all_frames/JPEGImages', video_name)
                mid_frame = frames_list[i]
                self.image_list.append([
                    os.path.join(base_dir, f'{int(mid_frame)-1:05d}.jpg'),
                    os.path.join(base_dir, f'{mid_frame}.jpg'),
                    os.path.join(base_dir, f'{int(mid_frame)+1:05d}.jpg'),
                ])
                self.image.append(
                    os.path.join(base_dir, f'{mid_frame}.jpg')
                )
                self.label.append(
                    os.path.join(data_dir, 'gts_person_noboard', video_name, mid_frame+'.png')
                )
                self.background_image.append(
                    os.path.join(base_dir, f'{frames_list[0]}.jpg'),
                )
                self.background_label.append(
                    os.path.join(data_dir, 'gts_person_noboard', video_name, frames_list[0]+'.png')
                )

    def size(self):
        return len(self.image_list)


class imagefile():
    def __init__(self, data_dir, list_file,
                 img_dir='binary_image', label_dir='binary_annos',
                 img_surfix='.jpg', label_surfix='.png'):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_surfix = img_surfix
        self.label_surfix = label_surfix
        with open(os.path.join(data_dir, list_file), "r") as f:
            image_list = [l.strip() for l in f.readlines()]
        self.image = [os.path.join(self.data_dir, self.img_dir, i + self.img_surfix)
                      for i in image_list]
        self.label = [os.path.join(self.data_dir, self.label_dir, i + self.label_surfix)
                      for i in image_list]

    def __len__(self):
        return len(self.image)


if __name__ == '__main__':
    data = imagefile('dataset/LV-MHP-v2', 'list/train.txt')
    print(data.image[0], data.label[0])
    data = imagefile('dataset/Supervisely', 'train.txt',
                     img_dir='SuperviselyImages', label_dir='SuperviselyMasks')
    print(data.image[0], data.label[0])
