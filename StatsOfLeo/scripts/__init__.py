#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

titles = []
notes = []

url = 'https://en.wikipedia.org/wiki/Leonardo_DiCaprio_filmography'
getRequest = requests.get(url)

html_doc = getRequest.text

soup = BeautifulSoup(html_doc, "html.parser")

for row in soup.findAll('table')[0].findAll('tr'):
    column = [x.text for x in row.find_all('th')]
    if len(column) >= 1:
        titles.append(column[0][0:len(column[0]) - 1])

titles = ["+".join(x.split(" ")) for x in titles]

for title in titles:
    url = 'http://www.omdbapi.com/?apikey=72bc447a&t=' + title

    getRequest = requests.get(url)

    json_data = getRequest.json()

    if len(json_data['Ratings']) > 0:
        ratingVal = json_data['Ratings'][0]['Value'][0:len(json_data['Ratings'][0]['Value']) - 3]
        notes.append(float(ratingVal))


plt.hist(notes)
plt.title("Film ratings with actor Leonardo DiCaprio")
plt.xlabel("Rate")
plt.ylabel("How many times")
plt.xticks([4, 5, 6, 7, 8, 9, 10])
plt.grid(True, axis='y')
plt.show()
