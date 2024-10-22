import os
import subprocess
import time
import signal
from multiprocessing import Pool, Manager, Lock
import sys
import argparse
import tqdm
import re
import decord
import csv
from datetime import datetime

def calculate_duration(start_time: str, end_time: str) -> str:
    # Define the format used for the input strings
    time_format = "%H:%M:%S.%f"
    
    # Convert the start and end time strings to datetime objects
    start = datetime.strptime(start_time, time_format)
    end = datetime.strptime(end_time, time_format)
    
    # Calculate the duration (difference between end and start)
    duration = end - start
    total_seconds = duration.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60  
    
    duration_str = f"{hours:02}:{minutes:02}:{seconds:06.3f}"
    
    return duration_str

def convert_duration_to_frames(duration: str, fps: int) -> int:
    # Use regex to parse the time format HH:MM:SS.mmm
    pattern = r"(\d+):(\d+):(\d+).(\d+)"
    match = re.match(pattern, duration)
    if not match:
        raise ValueError("Invalid duration format. It should be in HH:MM:SS.mmm")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    
    # Convert the entire duration into seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    # Calculate the total number of frames
    total_frames = int(total_seconds * fps)
    
    return total_frames

def calc_hash_value(video_id):
    hash_value = 0
    for i in range(len(video_id)):
        hash_value += ord(video_id[i])
    return hash_value

def run_command_with_timeout(command, timeout):
    if os.name != 'nt':
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    else:
        process = subprocess.Popen(command, shell=True)
    
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        
        time.sleep(0.5)
        
        if process.poll() is None:
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        
        return None
    
    return process.returncode


def process_videos(path, output_dir, video_cut_dict, shared_list, lock, visit_list, iidx):
    video_name = os.path.basename(path)
    if video_name.endswith(".csv") or video_name.endswith(".part") or video_name.endswith(".txt"):
        visit_list.append(0)
        pbar.n = len(visit_list)
        pbar.refresh()
        return
    
    with lock:
        cuda_id = shared_list.pop(0)
    video_base_name = video_name.split('.')[-2]
    try:
        video_path = path
        video = decord.VideoReader(video_path)
        fps = video.get_avg_fps()
        if "_" in video_base_name:
            bv_id = video_base_name.split("_")[0]
            video_part_str = video_base_name.split("_")[1]
        else:
            bv_id = video_base_name
            video_part_str = None
        if bv_id in video_cut_dict:
            for video_part in video_cut_dict[bv_id]:
                start_time = video_part["video_start"]
                end_time = video_part["video_end"]
                duration_time = calculate_duration(start_time, end_time)
                part_number = int(video_part["video_part"])
                clip_number = int(video_part["video_clip"])
                if video_part_str is not None:
                    v_part_number = int(video_part_str[1:])
                    if part_number != v_part_number:
                        continue
                clip_start = (clip_number-1) * 3600

                start_frame = (convert_duration_to_frames(start_time, fps) + clip_start) / fps
                
                output_name = os.path.join(output_dir, f"{video_base_name}_{clip_number}_{start_time}_{duration_time}.mp4")
                is_exist = os.path.exists(output_name)
                if is_exist:
                    continue
                command = f"CUDA_VISIBLE_DEVICES={cuda_id} ffmpeg -ss {start_frame}  -i {video_path} -t {duration_time} -c:v hevc_nvenc -an {output_name} -y"
                os.system(command)
    except:
        print('error!')
    with lock:
        shared_list.append(cuda_id)
    visit_list.append(0)
    pbar.n = len(visit_list)
    pbar.refresh()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--process-nums", type=int, default=1)
    parser.add_argument("--input-video-dir", type=str, default="", required=True)
    parser.add_argument("--csv-file", type=str, default="", required=True)
    parser.add_argument("--output-video-dir", type=str, default="", required=True)
    args = parser.parse_args()

    num_devices = args.num_devices
    pn = args.process_nums
    input_video_dir = args.input_video_dir
    output_video_dir = args.output_video_dir
    input_video_csv = args.csv_file

    video_cut_dict = {}
    with open(input_video_csv, 'r') as f:
        scene_list = csv.reader(f)
        for i, row in enumerate(scene_list):
            if i == 0:
                continue
            video_id = row[0].split("/")[-1]
            video_part = row[1]
            video_clip = row[2]
            video_start = row[3]
            video_end = row[4]
            video_smpl_file = row[5]
            if video_id not in video_cut_dict:
                video_cut_dict[video_id] = []
            video_cut_dict[video_id].append({
                "video_part": video_part,
                "video_clip": video_clip,
                "video_start": video_start,
                "video_end": video_end,
                "video_smpl_file": video_smpl_file
            })


    video_list = sorted(os.listdir(input_video_dir))
    video_list = [os.path.join(input_video_dir, video) for video in video_list]
    os.makedirs(output_video_dir, exist_ok=True)
    pbar = tqdm.tqdm(total=int(len(video_list)))
    with Manager() as manager:
        shared_list = manager.list()
        visit_list = manager.list()
        lock = manager.Lock()
        for i in range(pn):
            shared_list.append(i%num_devices)
        arguments = [(video, output_video_dir, video_cut_dict, shared_list, lock, visit_list, idx) for idx,video in enumerate(video_list)]
        with Pool(pn) as pool:
            pool.starmap(process_videos, arguments)
                                                  
