import os
import re
import time
from collections import Counter
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from transformers import pipeline
import wikipedia
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

console = Console()
CSV_FILENAME = "musicals_dataset.csv"

# --- NETWORK SETUP ---
# This makes connections significantly faster and adds automatic retries for timeouts/server errors
session = requests.Session()
retries = Retry(
    total=5, 
    backoff_factor=1, 
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)
TIMEOUT = 15 # Seconds to wait before throwing a timeout and triggering a retry

def sanitize_filename(name):
    """Sanitizes strings to be safe for OS directory and file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def load_processed_inventory():
    """Reads the CSV to figure out what individual songs have already been processed."""
    processed = set()
    if os.path.exists(CSV_FILENAME):
        try:
            df = pd.read_csv(CSV_FILENAME)
            for _, row in df.iterrows():
                processed.add((row['musical'], row['song_title']))
        except Exception:
            pass
    return processed

def append_to_csv(data_dict):
    """Appends a single row to the CSV immediately."""
    df = pd.DataFrame([data_dict])
    write_header = not os.path.exists(CSV_FILENAME)
    df.to_csv(CSV_FILENAME, mode='a', index=False, header=write_header, encoding="utf-8")

def get_wiki_metadata(musical_name):
    """Fetches Wikipedia summary and extracts basic metadata."""
    try:
        page = wikipedia.page(f"{musical_name} musical", auto_suggest=False)
        summary = page.summary
        
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', summary)
        year = year_match.group(1) if year_match else "Unknown"
        
        c_match = re.search(r'music by (.*?)[,\.]', summary, re.IGNORECASE)
        composer = c_match.group(1).strip() if c_match else "Unknown"
        
        l_match = re.search(r'lyrics by (.*?)[,\.]', summary, re.IGNORECASE)
        lyricist = l_match.group(1).strip() if l_match else "Unknown"
            
        return year, composer, lyricist
    except wikipedia.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', page.summary)
            return year_match.group(1) if year_match else "Unknown", "Unknown", "Unknown"
        except:
            return "Unknown", "Unknown", "Unknown"
    except Exception:
        return "Unknown", "Unknown", "Unknown"

def get_lexical_metrics(text, nlp):
    """Computes lexical richness, density, and most used non-stop words."""
    doc = nlp(text)
    words = [token.lemma_.lower() for token in doc if token.is_alpha]
    
    if not words:
        return 0, 0, []
    
    lexical_richness = len(set(words)) / len(words)
    
    content_words = [token for token in doc if token.is_alpha and token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
    lexical_density = len(content_words) / len(words)
    
    non_stop = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    most_used = [word for word, count in Counter(non_stop).most_common(5)]
    
    return lexical_richness, lexical_density, most_used

def get_emotions(text, emotion_pipeline):
    """Chunks text safely below the 512 token limit and averages the emotional scores."""
    words = text.split()
    if not words:
        return {}
    
    chunks = [' '.join(words[i:i+250]) for i in range(0, len(words), 250)]
    emotion_sums = Counter()
    
    for chunk in chunks:
        raw_results = emotion_pipeline(chunk, top_k=None, truncation=True, max_length=512)
        results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
        
        for score_dict in results:
            emotion_sums[score_dict['label']] += score_dict['score']
    
    num_chunks = len(chunks)
    avg_emotions = {f"emotion_{k}": round(v / num_chunks, 4) for k, v in emotion_sums.items()}
    
    return avg_emotions

def main():
    BASE_URL = "https://www.allmusicals.com"
    LETTERS = list("abcdefghijklmnopqrstuvwxyz") + ["19"]
    os.makedirs("data", exist_ok=True)
    
    console.print("[bold cyan]Loading NLP Models & Checking Progress...[/bold cyan]")
    nlp = spacy.load("en_core_web_sm")
    emotion_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        top_k=None
    )
    
    processed_inventory = load_processed_inventory()
    console.print(f"[bold green]Found {len(processed_inventory)} songs in CSV.[/bold green]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        letter_task = progress.add_task("[cyan]Overall Letters Progress", total=len(LETTERS))
        
        for letter in LETTERS:
            try:
                letter_url = f"{BASE_URL}/{letter}.htm"
                res = session.get(letter_url, timeout=TIMEOUT)
                if res.status_code != 200:
                    progress.advance(letter_task)
                    continue
            except Exception as e:
                console.print(f"[red]Failed to load letter {letter}: {e}[/red]")
                progress.advance(letter_task)
                continue
                
            soup = BeautifulSoup(res.text, "html.parser")
            musical_links = soup.select(".sym-album-list ul li a")
            
            musical_task = progress.add_task(f"[magenta]Musicals in '{letter.upper()}'", total=len(musical_links))
            
            for m_link in musical_links:
                musical_title = m_link.get_text(strip=True).replace(" Lyrics", "")
                safe_musical_title = sanitize_filename(musical_title)
                musical_dir = os.path.join("data", safe_musical_title)
                done_file = os.path.join(musical_dir, ".done")
                
                # Checkpoint: Skip network fetch entirely if musical was previously completed
                if os.path.exists(done_file):
                    progress.advance(musical_task)
                    continue
                
                musical_href = m_link.get('href')
                
                try:
                    m_res = session.get(BASE_URL + musical_href, timeout=TIMEOUT)
                    if m_res.status_code != 200:
                        progress.advance(musical_task)
                        continue
                except Exception as e:
                    console.print(f"[red]Failed to load musical {musical_title}: {e}[/red]")
                    progress.advance(musical_task)
                    continue
                    
                m_soup = BeautifulSoup(m_res.text, "html.parser")
                song_items = m_soup.select(".lyrics-list ol li")
                
                wiki_fetched = False
                year, composer, lyricist = "Unknown", "Unknown", "Unknown"
                current_act = "Act Unknown"
                
                song_task = progress.add_task(f"[yellow]Songs in '{musical_title[:20]}...'", total=len(song_items))
                
                for item in song_items:
                    if 'act' in item.get('class', []):
                        current_act = sanitize_filename(item.get_text(strip=True))
                        progress.advance(song_task)
                        continue
                        
                    a_tag = item.find('a')
                    if a_tag:
                        song_title = sanitize_filename(a_tag.get_text(strip=True))
                        
                        # Checkpoint: Skip specific song if already in CSV
                        if (musical_title, song_title) in processed_inventory:
                            progress.advance(song_task)
                            continue
                            
                        if not wiki_fetched:
                            year, composer, lyricist = get_wiki_metadata(musical_title)
                            wiki_fetched = True
                            
                        song_url = BASE_URL + a_tag['href']
                        save_dir = os.path.join(musical_dir, current_act)
                        os.makedirs(save_dir, exist_ok=True)
                        file_path = os.path.join(save_dir, f"{song_title}.txt")
                        
                        try:
                            song_res = session.get(song_url, timeout=TIMEOUT)
                            if song_res.status_code == 200:
                                s_soup = BeautifulSoup(song_res.text, "html.parser")
                                page_div = s_soup.find("div", id="page")
                                
                                if page_div:
                                    h2 = page_div.find("h2", class_="visible-print")
                                    if h2:
                                        h2.decompose()
                                    
                                    lyrics = page_div.get_text(separator="\n", strip=True)
                                    
                                    with open(file_path, "w", encoding="utf-8") as f:
                                        f.write(lyrics)
                                    
                                    richness, density, top_words = get_lexical_metrics(lyrics, nlp)
                                    emotions = get_emotions(lyrics, emotion_classifier)
                                    
                                    row_data = {
                                        "song_title": song_title,
                                        "musical": musical_title,
                                        "subsection": current_act,
                                        "year": year,
                                        "composer": composer,
                                        "lyricist": lyricist,
                                        "lexical_richness": round(richness, 4),
                                        "lexical_density": round(density, 4),
                                        "most_used_words": ", ".join(top_words)
                                    }
                                    row_data.update(emotions)
                                    
                                    append_to_csv(row_data)
                                    processed_inventory.add((musical_title, song_title))
                        except Exception as e:
                            # Catch any timeout/network error on the song and continue without crashing
                            console.print(f"[red]Error fetching song {song_title}: {e}[/red]")
                            
                        time.sleep(0.5)
                    progress.advance(song_task)
                
                # If we successfully made it through all songs in the musical, drop the .done file
                os.makedirs(musical_dir, exist_ok=True)
                with open(done_file, "w") as f:
                    f.write("done")
                
                progress.remove_task(song_task)
                progress.advance(musical_task)
                
            progress.remove_task(musical_task)
            progress.advance(letter_task)
            
    console.print("\n[bold green]Scraping Complete![/bold green]")

if __name__ == "__main__":
    main()
