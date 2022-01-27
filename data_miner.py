from re import T
from bs4 import BeautifulSoup

def fetch_page(url):
    import requests
    response = requests.get(url)
    print(f"Fetching: {url} ====> Response: ", response.status_code)
    return response.text

def save_words(words):
    with open('data/words.txt', 'a') as file:
        file.writelines('\n'.join(words))

def extract_words(soup):
    words = []

    span_mot = soup.find_all('span', class_='mot')
    
    # there were 2 classes mot and mot2
    for span in span_mot:
        words.extend(span.get_text().strip().split(" "))

    span_mot = soup.find_all('span', class_='mot2')
    
    for span in span_mot:
        words.extend(span.get_text().strip().split(" "))

    # could be in random order
    words.sort()

    print("Extracted till: ", words[-1])
    save_words(words)

def main():
    BASE_URL = 'https://www.bestwordlist.com/5letterwords.htm'

    # special case for 1
    text = fetch_page(BASE_URL)
    soup = BeautifulSoup(text, 'html.parser')
    extract_words(soup)

    for idx in range(2, 16):
        text = fetch_page(f"https://www.bestwordlist.com/5letterwordspage{idx}.htm")
        soup = BeautifulSoup(text, 'html.parser')
        extract_words(soup)
        
if __name__ == "__main__":
    main()
