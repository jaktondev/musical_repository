import os
import re
import pandas as pd
from collections import Counter
from transformers import pipeline
from rich.console import Console
from rich.progress import track

console = Console()

def sanitize_filename(name):
    """Sanitizes strings to reconstruct the OS directory paths."""
    return re.sub(r'[\\/*?:"<>|]', "", str(name)).strip()

def chunk_cleaned_lyrics(text, word_limit=80):
    """
    Strips stage directions, removes blank lines, and chunks 
    the text into slightly larger blocks for emotional context.
    """
    # 1. Strip out anything in brackets (e.g., [ALADDIN], [Spoken], [Chorus])
    text = re.sub(r'\[.*?\]', '', text)
    
    # 2. Split by lines and remove completely blank spaces/lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return []
        
    chunks = []
    current_chunk = ""
    
    # 3. Group lines into larger chunks (default ~80 words)
    for line in lines:
        if len(current_chunk.split()) + len(line.split()) < word_limit:
            current_chunk += " " + line
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def get_go_emotions(chunks, emotion_pipeline):
    """Runs chunks through GoEmotions and returns averaged scores."""
    if not chunks:
        return {}
        
    sums_28 = Counter()
    
    for chunk in chunks:
        res_28 = emotion_pipeline(chunk, top_k=None, truncation=True, max_length=512)
        res_28 = res_28[0] if isinstance(res_28[0], list) else res_28
        
        for score_dict in res_28:
            sums_28[score_dict['label']] += score_dict['score']
            
    num_chunks = len(chunks) if chunks else 1
    
    avg_emotions = {}
    for k, v in sums_28.items():
        avg_emotions[f"go_{k}"] = round(v / num_chunks, 4)
        
    return avg_emotions

def main():
    INPUT_CSV = "full_data.csv"
    OUTPUT_CSV = "go_emotions_data.csv"
    
    try:
        console.print(f"[bold cyan]Loading {INPUT_CSV}...[/bold cyan]")
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        console.print(f"[bold red]Error: Could not find {INPUT_CSV}.[/bold red]")
        return

    # Drop any old emotion columns to start fresh
    old_emotion_cols = [col for col in df.columns if col.startswith('emotion_') or col.startswith('go_') or col.startswith('hartmann_')]
    df = df.drop(columns=old_emotion_cols, errors='ignore')
    
    console.print("[bold cyan]Loading GoEmotions Model (28 classes)...[/bold cyan]")
    go_pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    console.print("[bold yellow]Running GoEmotions on cleaned lyrics...[/bold yellow]")
    
    new_rows = []
    
    for index, row in track(df.iterrows(), total=len(df), description="Processing songs..."):
        safe_musical = sanitize_filename(row['musical'])
        safe_sub = sanitize_filename(row['subsection'])
        safe_song = sanitize_filename(row['song_title'])
        
        path = os.path.join("data", safe_musical, safe_sub, f"{safe_song}.txt")
        row_dict = row.to_dict()
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                lyrics = f.read()
                
            # Use the new cleaning & chunking function
            chunks = chunk_cleaned_lyrics(lyrics, word_limit=80)
            emotion_scores = get_go_emotions(chunks, go_pipe)
            row_dict.update(emotion_scores)
            
        except FileNotFoundError:
            pass
            
        new_rows.append(row_dict)

    enriched_df = pd.DataFrame(new_rows)
    enriched_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    console.print(f"\n[bold green]Success! Created new dataset with cleaned text and 28 emotion classes saved to {OUTPUT_CSV}.[/bold green]")

if __name__ == "__main__":
    main()
