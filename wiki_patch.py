import os
import re
import time
import pandas as pd
import wikipedia
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# ==========================================
# CHANGE THIS TO 0 TO PROCESS ALL MUSICALS
# ==========================================
TEST_MODE = 0

def clean_infobox_data(td):
    """Cleans up the raw HTML inside a Wikipedia infobox cell."""
    # Remove citation brackets like [1], [a]
    for sup in td.find_all('sup'):
        sup.decompose()
        
    # Replace breaks and list items with commas so text doesn't mash together
    for br in td.find_all('br'):
        br.replace_with(', ')
    for li in td.find_all('li'):
        li.insert_after(', ')
        
    text = td.get_text(separator=' ', strip=True)
    
    # Clean up awkward double commas or trailing commas
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(', ')

def fetch_musical_info(musical):
    """Searches Wikipedia and parses the infobox and summary for metadata."""
    year, composer, lyricist = "Unknown", "Unknown", "Unknown"
    
    try:
        # 1. Search for the most accurate Wikipedia page title
        search_results = wikipedia.search(f"{musical} musical", results=3)
        if not search_results:
            search_results = wikipedia.search(musical, results=3)
            
        if not search_results:
            return year, composer, lyricist
            
        page_title = search_results[0]
        
        # 2. Fetch the page
        page = wikipedia.page(page_title, auto_suggest=False)
        html = page.html()
        summary = page.summary
        
        # 3. Parse the HTML Infobox
        soup = BeautifulSoup(html, "html.parser")
        infobox = soup.find('table', class_=re.compile('infobox'))
        
        if infobox:
            for row in infobox.find_all('tr'):
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    header = th.get_text(strip=True).lower()
                    value = clean_infobox_data(td)
                    
                    # Target specific rows
                    if 'music' in header and 'lyrics' in header:
                        if composer == "Unknown": composer = value
                        if lyricist == "Unknown": lyricist = value
                    elif 'music' in header or 'composer' in header:
                        if composer == "Unknown": composer = value
                    elif 'lyrics' in header or 'lyricist' in header:
                        if lyricist == "Unknown": lyricist = value
                    elif 'premiere' in header or 'date' in header or 'opened' in header or 'productions' in header:
                        if year == "Unknown":
                            y_match = re.search(r'\b(19\d{2}|20\d{2})\b', value)
                            if y_match:
                                year = y_match.group(1)
                                
        # 4. Fallbacks (If infobox was missing data, check the summary paragraph)
        if year == "Unknown":
            y_match = re.search(r'\b(19\d{2}|20\d{2})\b', summary)
            if y_match: year = y_match.group(1)
            
        if composer == "Unknown":
            c_match = re.search(r'(?:music|score)(?:\s+was)?(?:\s+composed)?\s+by\s+([A-Z][\w\s\.\-]+?)(?:,|\.|and|\bwith\b)', summary)
            if c_match: composer = c_match.group(1).strip()
            
        if lyricist == "Unknown":
            l_match = re.search(r'lyrics(?:\s+were)?(?:\s+written)?\s+by\s+([A-Z][\w\s\.\-]+?)(?:,|\.|and|\bwith\b)', summary)
            if l_match: lyricist = l_match.group(1).strip()
            
    except Exception:
        # Ignore DisambiguationErrors or PageErrors and just return Unknowns
        pass
        
    return year, composer, lyricist

def main():
    INPUT_CSV = "musicals_dataset.csv"
    OUTPUT_CSV = "full_data.csv"
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        console.print(f"[bold red]Error: Could not find {INPUT_CSV}.[/bold red]")
        return
        
    unique_musicals = df['musical'].unique().tolist()
    
    if TEST_MODE > 0:
        console.print(f"[bold yellow]--- RUNNING IN TEST MODE ({TEST_MODE} MUSICALS) ---[/bold yellow]")
        unique_musicals = unique_musicals[:TEST_MODE]
    else:
        console.print(f"[bold green]Processing all {len(unique_musicals)} unique musicals...[/bold green]")
        
    wiki_cache = {}
    
    for musical in track(unique_musicals, description="Fetching Wikipedia Data..."):
        year, comp, lyr = fetch_musical_info(musical)
        wiki_cache[musical] = {
            "year": year,
            "composer": comp,
            "lyricist": lyr
        }
        time.sleep(0.5) # Polite delay
        
    # If in test mode, print a visual table to verify results
    if TEST_MODE > 0:
        table = Table(title="Wikipedia Parsing Test Results")
        table.add_column("Musical", style="cyan", no_wrap=True)
        table.add_column("Year", style="magenta")
        table.add_column("Composer", style="green")
        table.add_column("Lyricist", style="yellow")
        
        for m, data in wiki_cache.items():
            table.add_row(m, data["year"], data["composer"], data["lyricist"])
            
        console.print(table)
        console.print("\n[bold yellow]Test complete. If the data looks good, change TEST_MODE = 0 in the script to run on all data and save to CSV.[/bold yellow]")
        return
        
    # If NOT in test mode, apply to dataframe and save
    console.print("\n[bold cyan]Merging data into original dataset...[/bold cyan]")
    df['year'] = df['musical'].map(lambda m: wiki_cache.get(m, {}).get('year', 'Unknown'))
    df['composer'] = df['musical'].map(lambda m: wiki_cache.get(m, {}).get('composer', 'Unknown'))
    df['lyricist'] = df['musical'].map(lambda m: wiki_cache.get(m, {}).get('lyricist', 'Unknown'))
    
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    console.print(f"[bold green]Success! Dataset saved as {OUTPUT_CSV}.[/bold green]")

if __name__ == "__main__":
    main()
