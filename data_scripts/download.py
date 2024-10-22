import os
import csv
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=str, default="", required=True)
    parser.add_argument("--yt-dlp-path", type=str, default="", required=True)
    parser.add_argument("--output-dir", type=str, default="", required=True)
    parser.add_argument("--download-nums", type=int, default=10, required=True)
    parser.add_argument("--cookies", type=str, default="", required=True)
    args = parser.parse_args()

    csv_file = args.csv_file
    cookie_file = args.cookies
    output_dir = args.output_dir
    download_nums = args.download_nums
    yt_dlp_path = args.yt_dlp_path
    os.makedirs(output_dir, exist_ok=True)

    csv_reader = csv.reader(open(csv_file, 'r'))
    for i, row in tqdm(enumerate(csv_reader)):
        if i == 0:
            continue
        if i > download_nums:
            break
        video_url = row[0]
        bv_id = row[0].split("/")[-1]
        output_path = os.path.join(output_dir, bv_id)
        if os.path.exists(output_path):
            continue
        
        command = f"{yt_dlp_path} -f 'bestvideo[height<=1920][ext=mp4]/mp4' --download-sections \"*00:00:00-00:05:00\" --cookies {cookie_file} -o \"{output_dir}/%(id)s.%(ext)s\"" + " https://www.bilibili.com/video/" + bv_id
        
        os.system(command)

