import cv2
from scipy.io import wavfile
import numpy as np 
import random
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import dlib
from skimage.transform import estimate_transform, warp

def load_video(path,size=(224,224),face_detector=None):
    cap = cv2.VideoCapture(path)
    rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    # i=0
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            cropped_image = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)

            ## detect face
            if face_detector is not None:
                list_face = face_detector(frame,1)
                if len(list_face) == 0:
                    print('warning: no detected face')
                    cropped_image = np.zeros(size +(3,))
                else:  
                    d = list_face[0].rect ## only use the first detected face (assume that each input image only contains one face)
                    left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
                    old_size = (right - left + bottom - top)/2
                    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
                    new_size = int(old_size*1.58)

                    # crop image
                    src_pts = np.array([[center[0]-new_size/2, center[1]-new_size/2], [center[0] - new_size/2, center[1]+new_size/2], [center[0]+new_size/2, center[1]-new_size/2]])
                    DST_PTS = np.array([[0,0], [0,size[1]-1 - 1], [size[0]-1 - 1, 0]])
                    tform = estimate_transform('similarity', src_pts, DST_PTS)
                    
                    cropped_image = warp(frame, tform.inverse, output_shape=size,preserve_range=True)
            frames.append(cropped_image)
            # plt.imsave('%d.png'%(i), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # i+=1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    # print("video recorded under %0.2f Hz; time length : %0.2f seconds"%(rate,len(frames)/rate))
    return np.array(frames),rate

def load_detector(prefix='./'):
    #---- load detectors
    detector_path = os.path.join(prefix, 'mmod_human_face_detector.dat')
    face_detector = dlib.cnn_face_detection_model_v1(
            detector_path)

    return face_detector

def loadWAV(path):
    sample_rate, audio  = wavfile.read(path)
    audiosize = audio.shape[0]
    feats = []
    feats.append(audio)
    feat = np.stack(feats,axis=0).squeeze()
    return feat


def slice_audio(aframes, half_window=400, vf_index=[],apv_rate=16000/25):
    a_index = [[range(int(i*apv_rate) - half_window,int(i*apv_rate) + half_window)] for i in vf_index]
    audio = [aframes[i] for i in a_index]
    return np.array(audio)


# def sample_data(root_path_video,root_path_audio,id,file_name):
#     path_video = os.path.join(root_path_video,id,file_name)
#     path_audio = os.path.join(root_path_audio,id,file_name)

#     vframe,vrate = load_video(path_video)
#     aframe = loadWAV(path_audio)
#     arate = 16000
#     pav_rate = (arate - 1)/(vrate - 1)
    

#     pass



class VoxcelebDataset(Dataset):
    def __init__(self,root_video_path='/home/ubuntu/dev/mp4',root_audio_path='/home/ubuntu/data/voxceleb2',video_file='./count_frame.txt',resl = (128,128), max_len = 75,face_detector=False,step=1,half_window=1200, **kwargs):
        
        self.root_video = root_video_path
        self.root_audio = root_audio_path
        # self.data_dict = { x: {} for x in os.listdir(self.root_video) if len(os.listdir(os.path.join(self.root_video,x)))>0}
        self.resl = resl
        self.max_len = max_len
        self.step=step
        self.half_window = half_window

        ## add video_list for data
        self.video_dict = {}
        with open(video_file,'r+') as f:
            for line in f.readlines():
                if line.split(' ')[0] not in self.video_dict:
                    self.video_dict[line.split(' ')[0]] = line.split(' ')[1]
                else:
                    continue

        # names = [x for x in os.listdir(self.root_video) if len(os.listdir(os.path.join(self.root_video,x)))>0]
        self.data_list = []

        # func = lambda root,id: [i for i in os.listdir(os.path.join(root,id)) if os.path.isdir(os.path.join(root,id,i))]

        # for key in self.data_dict:
        #     for j in func(self.root_video,key):
        #         v_filename = [name for name in os.listdir(os.path.join(self.root_video,key,j)) \
        #             if name.endswith('.mp4') and os.path.isfile(os.path.join(self.root_audio,key,j,name.replace('.mp4','.wav')))]

        #         a_filename = [v.replace('.mp4','.wav') for v in v_filename ]

        #         self.data_dict[key][j] = [(os.path.join(self.root_video,key,j,v_filename[i]),os.path.join(self.root_audio,key,j,a_filename[i])) for i in range(len(v_filename))]
        #         self.data_list+= [(os.path.join(self.root_video,key,j,v_filename[i]),os.path.join(self.root_audio,key,j,a_filename[i])) for i in range(len(v_filename))]


        ## add code for data list
        for vid_link,l in self.video_dict.items():
            if int(l) > (self.max_len + 6) and os.path.isfile(os.path.join(self.root_audio,'/'.join(vid_link.split('/')[5:]).replace('.mp4','.wav'))):
                self.data_list.append((vid_link,os.path.join(self.root_audio,'/'.join(vid_link.split('/')[5:]).replace('.mp4','.wav'))))
        
        # remove_items = []
        # for key in self.data_dict:
        #     for j in self.data_dict[key]:
        #         if self.data_dict[key][j] == None or len(self.data_dict[key][j])==0:
        #             remove_items.append((key,j))
        # for key,j in remove_items:
        #     del self.data_dict[key][j]
        # ### Read Training Files...

        # remove_scene = []
        # for key in self.data_dict:
        #     if len(self.data_dict[key]) == 0:
        #         remove_scene.append(key)
        
        # for scene in remove_scene:
        #     del self.data_dict[scene]

        # self.identity = list(self.data_dict.keys())

        random.seed(1000) ##set random seed
        self.data_list = random.sample(self.data_list,1000)
        # self.length = []
        # for v in self.data_list:
        #     if len(v) ==2:
        #         frames,_ = load_video(v[0],size=resl)
        #         l = len(frames)
        #         self.length.append(l)
        #     else:
        #         self.length.append(-1)
        if face_detector:
            self.face_detector = load_detector()
        else:
            self.face_detector = None



    def __len__(self):
        return len(self.data_list)

    def update_resl(self,resl):
        self.resl = resl

    def update_step(self,step):
        self.step = step


    def __getitem__(self,idx):
        # id = self.identity[idx]
        # v = random.choice(list(self.data_dict[id].keys()))
        # while len(self.data_dict[id][v])==0:
        #     del self.data_dict[id][v]
        
        #     v = random.choice(list(self.data_dict[id].keys()))
            
        # source  = random.choice(self.data_dict[id][v])
        
        source = self.data_list[idx]
        video_source,_ = load_video(source[0],self.resl,self.face_detector)

        startframe = len(video_source) - self.max_len
        # while startframe <=1:
        #     print("The file is too short %d frames, at link %s" %(len(video_source),source[0]))
        #     # self.data_dict[id][v].remove(source)
        #     # source  = random.choice(self.data_dict[id][v])
        #     # video_source,_ = load_video(source[0],self.resl)
        #     # startframe = len(video_source) - self.max_len
        #     del self.data_list[idx]
        #     source = self.data_list[idx]
        #     video_source,_ = load_video(source[0],self.resl,self.face_detector)

        #     startframe = len(video_source) - self.max_len
            
        startframe = random.sample(range(startframe + 1),2)
        video_source_index = range(startframe[0] + 2 , startframe[0] + self.max_len - 2,self.step)
        video_target_index = range(startframe[1] + 2 , startframe[1] + self.max_len - 2,self.step)


        audio_source = loadWAV(source[1])

        target_audio = slice_audio(audio_source, half_window=self.half_window, vf_index=video_target_index,apv_rate=(len(audio_source)-1)/(len(video_source)-1))
        source_audio = slice_audio(audio_source, half_window=self.half_window, vf_index=video_source_index,apv_rate=(len(audio_source)-1)/(len(video_source)-1))
        

        source_video = video_source[[i for i in video_source_index]]
        target_video = video_source[[i for i in video_target_index]]
    

        return (np.array(source_video).swapaxes(2,3).swapaxes(1,2).astype(np.float32)/127.5 -1,\
                    np.array(source_audio).astype(np.float32),\
                    np.array(target_video).swapaxes(2,3).swapaxes(1,2).astype(np.float32)/127.5 -1,\
                    np.array(target_audio).astype(np.float32))





        


        

    


