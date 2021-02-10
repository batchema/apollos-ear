import os
import subprocess
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pytube import YouTube
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

PLAYLIST_LINKS = {
    "rumba": ["https://www.youtube.com/playlist?list=PLo_Xxb-8Kxp1OqEGzInLf0tvGO-iPEjla",
              "https://www.youtube.com/watch?v=r9YyPUrre9k&list=PLof-qyPO5RmfbxLarEMM_8hAjPtpkJ9lV",
              "https://www.youtube.com/watch?v=p0WiuzP-uQY&list=PLoMTMPMXFsMAPMKT66wNHvFj6PvSHVtYH"],
    "afrobeat": ["https://www.youtube.com/watch?v=Ecl8Aod0Tl0&list=PLX9U3Rv7Wy7X55IaMXkYHai_aoexDiGky"],
    "coupe_decale": ["https://www.youtube.com/watch?v=CbRH5KT2q9A&list=PLt8Fq_KSyWQtN2-4F_IONQvgcLNfTmRS-"]
}

LAST_LINK = {
    "hip_hop": ["https://www.youtube.com/watch?v=JFm7YDVlqnI&list=PL-FVH5VWgRPHNz24zZ5_FLHQWoidN6O1d"]
}

BROWSER_PATH = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'
SELENIUM_CHROME_EXECUTABLE_PATH = 'chromedriver'


def initiate_selenium():
    """
    Initiate Selenium Browser Automator
    """
    options = Options()
    options.binary_location = BROWSER_PATH
    driver = webdriver.Chrome(chrome_options=options, executable_path=SELENIUM_CHROME_EXECUTABLE_PATH)
    return driver


def extract_playlists_links(playlist_list_dict):
    """
    Extract individual video links from dictionary of lists of playlist links {a: [l0, l1, ..]}
    Arguments:
        playlist_list_dict: Dictionary of list of playlist links (String)
    """
    driver = initiate_selenium()
    results = {}
    # loop over genres and acquire proportional number of videos by subgenre playlists
    for playlist_list in playlist_list_dict:
        links = []
        playlists = playlist_list_dict[playlist_list]
        num_per_playlist = 100 // len(playlists) + 10 if len(playlists) > 1 else 100
        for link in playlists:
            driver.get(link)
            wait = WebDriverWait(driver, 3)
            visible = ec.visibility_of_element_located
            wait.until(visible((By.ID, "thumbnail")))
            videos = driver.find_elements_by_xpath('//*[@id="thumbnail"]')
            num_to_get = num_per_playlist
            if len(videos) < num_per_playlist:
                delta = num_per_playlist - len(videos)
                num_to_get = len(videos)
                num_per_playlist += delta
            for i in range(num_to_get):
                curr = videos[i]
                link = curr.get_attribute('href')
                if link:
                    img = curr.find_element_by_id('img')
                    if 'meh_mini' not in str(img.get_attribute('src')):
                        links.append(link.split('&')[0])

        results[playlist_list] = links
    return results


def clean_links_list(links):
    """
    Collect list of unbroken Youtube video links from list
    Arguments:
        links: List of Youtube video links
    Returns:
        List containing only working links
    """
    driver = initiate_selenium()
    cleans = set()
    for link in links:
        driver.get(link)
        wait = WebDriverWait(driver, 3)
        visible = ec.visibility_of_element_located

        # play the video
        wait.until(visible((By.ID, "video-title")) or visible((By.ID, "reason")))
        if driver.find_element_by_id("video-title"):
            cleans.add(link)
    return cleans


def acquire_video_audios(genres, limit=50):
    """
    Collect audio of youtube videos
    Arguments:
        genres: dictionary of (genre_title, playlist links)
        limit: The maximum number of audio files to acquire per genre
    Returns:
        List containing only working links
    """
    i = 0
    while i < limit:
        for genre in genres:
            for link in genres[genre]:
                try:
                    yt = YouTube(link)
                    yt.streams.filter(only_audio=True)[0].download(f'data/genres_added/{genre}')
                    i += 1
                except Exception:
                    print(link)


def convert_to_wav(folder_path, target_folder_name="cleans"):
    """
    Convert every audio file in a folder to wav format. Ensures that
    every subdirectory has a counterpart in target folder.
    Arguments:
        folder_path: path to top most folder
        target_folder_name: name of target folder. Will be sub-folder of
        folder_path's folder
    """
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(folder_path)):
        # ensure we're processing a genre sub-folder level
        if dirpath is not folder_path:
            sub_folder_name = dirpath.split('/')[-1]
            os.mkdir(f'{dirpath}/{target_folder_name}')

            # process all audio files in genre sub-dir
            i = 1
            for f in filenames:
                file_path = f'"{dirpath}/{f}"'
                new_filename = f'{dirpath}/{target_folder_name}/{sub_folder_name}000{i}.wav'
                ffmpeg = f'ffmpeg -i {file_path} {new_filename}'
                subprocess.run(ffmpeg, shell=True)
                i += 1


def select_snippets(folder_path, limit):
    """
    Collect 30 second snippets from audio files
    Arguments:
        folder_path: String path to the parent folder of genre folders
        limit: Integer number of snippets to collect
    """
    files = os.listdir(folder_path)
    genres = [os.path.join(folder_path, file) for file in files if os.path.isdir(os.path.join(folder_path, file))]
    os.mkdir(f'{folder_path}/cleans')
    for folder in genres:
        i = 0
        genre_name = folder.split('/')[-1]
        os.mkdir(f'{folder_path}/cleans/{genre_name}')
        files = os.listdir(folder)
        for file in os.listdir(folder):
            if i < limit:
                if file[-4:] == '.wav':
                    file_path = f'"{folder}/{file}"'
                    # Remove silence
                    remove_silence(file_path[1:-1])
                    # Extract snippet at random intervals, one close to the beginning, one close to the end
                    duration = get_duration(file_path)
                    target1 = f'{folder_path}/cleans/{genre_name}/{genre_name}00000{i}.wav'
                    i += 1
                    start1 = random.randint(10, duration - 10)
                    ffmpeg = f'ffmpeg -i {file_path} -ss {start1} -t 30 {target1}'
                    subprocess.run(ffmpeg, shell=True)

        # Make sure there are enough snippets per genre
        if i != limit:
            while i != limit:
                file = files[random.randint(0, len(files) - 1)]
                file_path = f'"{folder}/{file}"'
                start2 = random.randint(10, get_duration(file_path) - 10)
                target2 = f'{folder_path}/cleans/{genre_name}/{genre_name}000{i}.wav'
                i += 1
                ffmpeg = f'ffmpeg -i {file_path} -ss {start2} -t 30 {target2}'
                subprocess.run(ffmpeg, shell=True)


def get_duration(audio_path):
    """
    Get duration of audio file
    Arguments:
        audio_path: String path to the audio file
    Returns:
        The duration of the audio in integers
    """
    ffprobe = f'ffprobe -i {audio_path} -show_format -v quiet | sed -n \'s/duration=//p\''
    cmd = subprocess.run(ffprobe, shell=True, stdout=subprocess.PIPE)
    return int(float(cmd.stdout))


def remove_silence(audio_path):
    """
    Remove silence from audio file in place
    Arguments:
        audio_path: String path to the audio file
    Returns:
    """
    folder = os.path.dirname(audio_path)
    temp = f'{folder}/temp.wav'
    # ffmpeg = f'ffmpeg -i {audio_path} -af silenceremove=start_periods=1:start_silence=0.1:start_threshold=-96dB,areverse' \
    #          f',silenceremove=start_periods=1:start_silence=0.1:start_threshold=-96dB,areverse {temp}'
    ffmpeg = f'ffmpeg -i {audio_path} -af silenceremove=start_periods=1:stop_periods=1:detection=peak {temp}'

    subprocess.run(ffmpeg, shell=True)
    subprocess.run(f'mv -f {temp} {audio_path}', shell=True)
