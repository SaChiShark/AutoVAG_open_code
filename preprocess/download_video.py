from pytubefix import Playlist
import os
import tqdm
base_dir = '../data/courses'
playlists = [
    {
        'url':'https://www.youtube.com/playlist?list=PLOAQYZPRn2V4jYwTGKUH4YaU6NE6VROZX',
        'name':'ADL'
    },
    {
        'url':'https://www.youtube.com/playlist?list=PLJV_el3uVTsPM2mM-OQzJXziCGJa8nJL8',
        'name':'ML2022'
    },
    {
        'url':'https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J',
        'name':'ML2021'
    },
    {
        'url':'https://www.youtube.com/playlist?list=PLJV_el3uVTsPz6CTopeRp2L2t4aL_KgiI',
        'name':'Gai'
    }
]
for playlist in playlists:
    p = Playlist(playlist['url'])
    name = playlist['name']
    os.makedirs(f'{base_dir}/{name}/videoes',exist_ok=True)
    os.makedirs(f'{base_dir}/{name}/audios',exist_ok=True)
    for video in tqdm.tqdm(p.videos):
            safe_title = video.title.replace('|', '').replace('/', '').replace('\\', '').replace('.', '').replace(':', '')
            if 'QA' in safe_title or 'TA' in safe_title or 'Homework' in safe_title or '助教' in safe_title or 'Guest' in safe_title:
                continue
            video.streams.get_highest_resolution(progressive=False).download(output_path = f'{base_dir}/{name}/videoes',filename=f'{safe_title}.mp4')
            audio_stream = video.streams.filter(only_audio=True).order_by('abr').desc().first().download(output_path = f'{base_dir}/{name}/audios', filename=f'{safe_title}.mp3')
