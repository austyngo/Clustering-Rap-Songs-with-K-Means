import lyricsgenius
import pandas as pd
import time
from urllib.request import Request, urlopen, quote
from bs4 import BeautifulSoup
import requests
import json
import os

import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

#Scrape ranker.com for list of top rappers
def getListFromRanker_Selenium(url):

    browser = webdriver.Chrome(r"C:\Users\austi\Downloads\chromedriver_win32\chromedriver.exe")
    browser.get(url)
    time.sleep(1)
    elem = browser.find_element_by_tag_name("body")
    no_of_pagedowns = 100

    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.4)
        no_of_pagedowns-=1

    post_elems = browser.find_elements_by_class_name("listItem__data")
    artist_names = [p.find_element_by_class_name("listItem__title").text for p in post_elems[:-1]]
    browser.quit()
    return artist_names

def main():

    url = "https://www.ranker.com/crowdranked-list/the-greatest-rappers-of-all-time"
    artist_names = getListFromRanker_Selenium(url)
    print(artist_names)

    return artist_names

if __name__ == '__main__':
    artist_names = main()

token = "<use your Genius API token>"
api = lyricsgenius.Genius(token, remove_section_headers=True,
                 skip_non_songs=True, excluded_terms=["Remix", "Edit"], timeout = 10000, sleep_time=0.4) #exclude songs that include the terms "remix" and "edit"


#Starting the song search for each artist
query_number = 0
time1 = time.time()
tracks = []

for artist in artist_names:
    query_number += 1
    #Empty lists for information for each song
    #I edited the LyricsGenius package code to also retrieve record label information for each song
    artists, titles, albums, years, lyrics, label, writers = [], [], [], [], [], [], []
    print('\nQuery number:', query_number)
    #Search for max_songs per artist = n and sort them by popularity
    artist = api.search_artist(artist, max_songs = 150, sort='popularity')
    songs = artist.songs
    song_number = 0
    #Append information for each song in the previously created lists
    for song in songs:
        if song is not None:
            song_number += 1
            print('\nSong number:', song_number)
            print('\nNow adding: Artist')
            artists.append(song.artist)
            print('Now adding: Title')
            try:
                titles.append(song.title)
            except TypeError:
                title.append('NA')
            print('Now adding: Album')
            try:
                albums.append(song.album)
            except TypeError:
                albums.append('NA')
            print('Now adding: Year')
            try:
                years.append(song.year[0:4])
            except TypeError:
                years.append('NA')
            print('Now adding: Lyrics')
            try:
                lyrics.append(song.lyrics)
            except TypeError:
                lyrics.append('NA')
            print('Now adding: label')
            label.append(song.label)
            print('Now adding: Writers')
            try:
                writers.append(song.writer_artists)
            except TypeError:
                writers.append('NA')
                #Create a dataframe for each song's information and add it to a song list
    df = pd.DataFrame({'artist':artists, 'title':titles, 'album':albums, 'year':years, 'lyrics':lyrics, 'label':label, 'writers': writers})
    tracks.append(df)
    time2 = time.time()
    print('\nQuery', query_number, 'finished in', round(time2-time1,2), 'seconds.')

time3 = time.time()
#Create final song list
tracklist = pd.concat(tracks, ignore_index=True)
print('\nFinal tracklist of', query_number, 'artists finished in', round(time3 + time2,2), 'seconds.')

print(tracklist)
tracklist.to_csv('rap_songs.csv', encoding = 'utf-8', index=False)

#inspired by https://github.com/johnwmillr/LyricsGenius
