from bs4 import BeautifulSoup
import requests

# utilility script to load contents from Wikipedia and store them as .txt file

urls = [
    'https://en.wikipedia.org/wiki/C_language',
    'https://en.wikipedia.org/wiki/C++'
]

def fetch_wikipedia_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = ''
    for p in soup.find_all('p'):
        content += p.get_text() + '\n'
    return content

c_language_content = fetch_wikipedia_content(urls[0])
c_plus_plus_content = fetch_wikipedia_content(urls[1])

with open('c_language.txt', 'w', encoding='utf-8') as file:
    file.write(c_language_content)
with open('c_plus_plus_language.txt', 'w', encoding='utf-8') as file:
    file.write(c_plus_plus_content)