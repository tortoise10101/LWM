import json
import time

# crate extract_frames.sh in dataset/frames
# execute ./extract_frames.sh to fetch frames from youtube

# installations youtube-dl for ubuntu
# pip install youtube-dl
# hash youtube-dl

# ffmpeg -ss "00:00:10" -i $(youtube-dl -f 22 --get-url "https://www.youtube.com/watch?v=HeaX1dOxUec") -vframes 1 -q:v 2 tmp.jpgx

datatype = 'train'
with open(f'dataset/captions/{datatype}.json') as fp:
    js = json.load(fp)

print(len(js))
long_id = []
for i in js.keys():
    if len(js[i]['sentences']) >= 5:
        long_id.append(i)


cmds = []
for id in long_id:
    url = "https://www.youtube.com/watch?v=" + id[2:]
    indx = 0
    for timestamp in js[id]['timestamps']:
        mean = int((timestamp[0] + timestamp[1]) / 2)
        st = time.strftime('%H:%M:%S', time.gmtime(mean))
        cmds.append(
            f'ffmpeg -ss "{st}" -i $(youtube-dl -f bestvideo --get-url "{url}") -vframes 1 -q:v 2 {id}-{indx}.jpg')
            # ffmpeg -ss "00:00:10" -i $(youtube-dl -f 22 --get-url "https://www.youtube.com/watch?v=HeaX1dOxUec") -vframes 1 -q:v 2 tmp.jpgx
        indx += 1

'''
with open('dataset/frames/extract_frames.sh', 'w') as fp:
    fp.write('#!/bin/bash\n')
    for i in cmds:
        fp.write(i + '\n')
print("Generate loader script in dataset/frames/extract_frames.sh")
'''

# for multi
import numpy as np
cnt = 0
for i in np.array_split(cmds, 12):
    cnt += 1
    with open(f'dataset/frames/extract_frames{cnt}.sh', 'w') as fp:
        fp.write('#!/bin/bash\n')
        for j in i:
            fp.write(j + '\n')
