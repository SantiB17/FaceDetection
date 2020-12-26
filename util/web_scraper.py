import requests
from bs4 import BeautifulSoup
import os


def imagedown(url, folder):
    try:
        os.mkdir(os.path.join(os.getcwd(), folder))
    except:
        pass
    os.chdir(os.path.join(os.getcwd(), folder))
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img',{"src":True})
    for image in images:
        name = image['alt']
        link = image['src']
        with open(name.replace(' ', '-').replace('/', '').replace('"', '').replace('*', '').replace('?', '') + '.jpg', 'wb') as f:
            try:
                im = requests.get(link)
                f.write(im.content)
                print('Writing: ', name)
            except:
                pass


imagedown('https://www.gettyimages.com/photos/kanye-west?family=editorial&groupbyevent=true&numberofpeople=none,one&page=15&phrase=kanye%20west&sort=mostpopular', 'kanye')
