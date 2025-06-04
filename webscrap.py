from icrawler.builtin import GoogleImageCrawler
import os

def download_pokemon_images(pokemon_name, num_images=50):
    save_dir = f'dataset/{pokemon_name.lower()}'
    os.makedirs(save_dir, exist_ok=True)

    google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    google_crawler.crawl(keyword=f'{pokemon_name} pokemon', max_num=num_images)

pokemon_list = ["Pikachu", "Charmander", "Squirtle", "Bulbasaur", "Eevee"]

for pokemon in pokemon_list:
    download_pokemon_images(pokemon, num_images=200)

