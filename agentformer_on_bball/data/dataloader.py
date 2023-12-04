from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from .bball_split import get_bball_split_small, get_bball_split_medium, get_bball_split_large
from utils.utils import print_log


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training', set_type='small'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        self.set_type = set_type
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred           
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'eth_sub', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy            
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        elif parser.dataset in {'basketball'}:
            data_root = parser.data_root_basketball
            cfg_dir = parser.cfg_dir
            if 'small' in cfg_dir:
                seq_train, seq_val, seq_test = get_bball_split_small(parser.dataset)
            elif 'medium' in cfg_dir:
                seq_train, seq_val, seq_test = get_bball_split_medium(parser.dataset)
            elif 'large' in cfg_dir:
                seq_train, seq_val, seq_test = get_bball_split_large(parser.dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root
        
        if self.set_type=='small':
            self.batch_size=32
        elif self.set_type=='medium':
            self.batch_size=64
        elif self.set_type=='large':
            self.batch_size=128
        else:
            self.batch_size=None
        
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        self.past_frames_list = []
        
        for ss, seq_name in enumerate(self.sequence_to_load):
            print_log("loading sequence {} ...".format(seq_name), log=log)
            label_path = f'{data_root}/{seq_name}.txt'
            
            if parser.dataset=='basketball':
                if not(os.path.isfile(label_path)):
                    print('Skipping %s due to unavailable data'%seq_name)
                    continue
                        
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)
            min_past_frames = parser.min_past_frames
            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            
        
            self.num_total_samples += num_seq_samples
            
            self.past_frames_list.append(int(min_past_frames))
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
            
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)
    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.past_frames_list[seq_index] - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.batch_size is None:
            if self.index >= self.num_total_samples:
                self.index = 0      # reset
                return True
            else:
                return False
        else:
            if self.index >= self.batch_size:
                self.index = 0      # reset
                return True
            else:
                return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()
