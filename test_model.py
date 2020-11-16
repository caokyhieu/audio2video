import torch
import torch.nn as nn
import numpy as np
import os
import glob
from model import FinalModel,write_video
import subprocess
from scipy.io import wavfile
from video_loader import load_video,loadWAV,slice_audio


def write_test_video(path_model,path_video,path_audio,path_target_audio,style_dim=32,resl=(128,128),step=1,half_window=400):
    video_source,_ = load_video(path_video,resl)
    audio_source =  loadWAV(path_audio)
    audio_target = loadWAV(path_target_audio)
    if len(audio_target) > len(audio_source):
        print('Audio target is longer than source.Successful')
    else:
        print('Audio target is shorter than source.Failed')
        return

    truncate_frame = 2
    batch = 24

    video_source_index = range( truncate_frame , (len(video_source) - 2*truncate_frame)//batch * batch + truncate_frame,step)
    # trunc_tail = len(video_source) - len(video_source_index) - truncate_frame
    apv_rate = (len(audio_source)-1)/(len(video_source)-1)
    source_audio = slice_audio(audio_source, half_window=half_window, vf_index=video_source_index,apv_rate=apv_rate)
    target_audio = slice_audio(audio_target, half_window=half_window, vf_index=video_source_index,apv_rate=apv_rate)
    source_video = video_source[[i for i in video_source_index]]

    video_data,audio_data,audio_target_data = (np.array(source_video).swapaxes(2,3).swapaxes(1,2).astype(np.float32)/127.5 -1,np.array(source_audio).astype(np.float32),np.array(target_audio).astype(np.float32))
    model = FinalModel(size=128,style_dim=style_dim)
    # device = torch.device('cpu')
    # model.to(device)
    model.cuda()

    ## load model
    # checkpoint = torch.load(path_model,map_location = device )
    checkpoint = torch.load(path_model)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print('Load model successfully')
    
    gen_list = []
    gen_target_list = []
    for idx in range(0,len(video_data),batch):
        
        video_input = torch.from_numpy(video_data[idx:idx+batch]).cuda()
        audio_input = torch.from_numpy(audio_data[idx:idx+batch]).cuda()
        audio_target_input = torch.from_numpy(audio_target_data[idx:idx+batch]).cuda()
        print('Load data successfully')
       

        gen_images,_ = model(video_input,audio_input)
        gen_images = gen_images.detach().cpu().numpy().swapaxes(1,2).swapaxes(2,3)
        print('Get gen images successfully')
        gen_list.append(gen_images)

        gen_target_images,_ = model(video_input,audio_target_input)
        gen_target_images = gen_target_images.detach().cpu().numpy().swapaxes(1,2).swapaxes(2,3)
        gen_target_list.append(gen_target_images)
    gen_list = np.concatenate(gen_list,axis=0)
    gen_target_list = np.concatenate(gen_target_list,axis=0)

    write_video(gen_list,path='./test_result/demo.avi',size=(128,128))
    write_video(gen_target_list,path='./test_result/demo_target.avi',size=(128,128))
    print('write video successfully')
    samplerate = 16000
    wavfile.write("./test_result/demo.wav", samplerate, audio_source[int(apv_rate*video_source_index[0]) - half_window:int(apv_rate*video_source_index[-1]) + half_window])
    print('write source audio successfully')
    wavfile.write("./test_result/demo_target.wav", samplerate, audio_target[int(apv_rate*video_source_index[0]) - half_window:int(apv_rate*video_source_index[-1]) + half_window])
    print('write target audio successfully')
    ## concatenate video and audio
    subprocess.call('ffmpeg -i ./test_result/demo.avi -i ./test_result/demo.wav -shortest -c:v copy -c:a aac ./test_result/output.avi', shell=True)
    subprocess.call('ffmpeg -i ./test_result/demo_target.avi -i ./test_result/demo_target.wav -shortest -c:v copy -c:a aac ./test_result/output_target.avi', shell=True)


    print('concatenate successfully')
    pass

if __name__ =='__main__':
    write_test_video('./save_model/model_208.pth',path_video='/home/ubuntu/dev/mp4/id00397/AtjMYQ6XsWo/00001.mp4',path_audio='/home/ubuntu/data/voxceleb2/id00397/AtjMYQ6XsWo/00001.wav',path_target_audio='/home/ubuntu/data/voxceleb2/id00397/CuhtLlTmt2E/00002.wav',style_dim=32,resl=(128,128),step=1,half_window=400)
