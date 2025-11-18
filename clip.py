import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Directory containing videos
video_dir = './humanData'
outdir = './humanData_clip'

print(len(os.listdir("humanData_clip")))
print(len(os.listdir("humanData_dwpose_clip")))
exit(0)

# video_dir = './humanData_dwpose'
# outdir = './humanData_dwpose_clip'

os.makedirs(outdir, exist_ok=True)

# Duration of each clip
clip_duration = 5

# Loop over all mp4 files in the directory
for filename in sorted(os.listdir(video_dir)):
    if filename.endswith('.mp4'):
        # Full path to the video file
        video_file = os.path.join(video_dir, filename)

        # Get the duration of the video
        total_duration = int(float(os.popen(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_file}").read()))

        # Calculate the number of clips
        clips = total_duration // clip_duration

        # Loop over the clips
        if clips == 0:
            output_file = f"{outdir}/{filename[:-4]}_clip_{0}.mp4"
            if os.path.exists(output_file):
                continue
            os.system(f"ffmpeg -i {video_file} -ss {0} {output_file}")
        else:
            for i in range(clips):
                # Calculate the start and end time
                start_time = i * clip_duration
                end_time = (i + 1) * clip_duration

                # Output file name
                output_file = f"{outdir}/{filename[:-4]}_clip_{i}.mp4"
                if os.path.exists(output_file):
                    continue
                os.system(f"ffmpeg -i {video_file} -ss {start_time} -t 5 {output_file}")
            
            # Use moviepy to extract the clip
            # ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=output_file)

def generate_csv(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    output_file = "/root/autodl-tmp/human_dataset/humanData.csv"
    f = open(output_file, "w")
    writer = csv.writer(f)
    video_dir = "/root/autodl-tmp/human_dataset/humanData_clip"
    pose_dir = "/root/autodl-tmp/human_dataset/humanData_dwpose_clip"
    
    video_files = sorted(list([os.path.join(video_dir, v) for v in os.listdir(video_dir)]))
    post_files = sorted(list([os.path.join(pose_dir, v) for v in os.listdir(pose_dir)]))
    video_lengths = [100 for _ in range(len(video_files))]
    outputs = ["a video" for _ in range(len(video_files))]

    # save results
    result = list(zip(video_files, outputs, video_lengths, post_files))
    for t in result:
        writer.writerow(t)

    f.close()