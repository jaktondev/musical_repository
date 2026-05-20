import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import numpy as np

# --- CONFIGURATIONS ---
st.set_page_config(page_title="Musical Lyrics Explorer", layout="wide")

PROJECT_NAME = "🎭 Musical Repository"
FOOTER_MESSAGE = "Built by Juipo/Jaktondev/Jorge Jacuinde for an ENES Morelia Project, Copyright by Juipo"

# --- PERMANENT EMOTION COLOR MAP ---
EMOTION_COLORS = {
    'Admiration': '#2E8B57', 'Amusement': '#FFA500', 'Anger': '#DC143C', 
    'Annoyance': '#B22222', 'Approval': '#3CB371', 'Caring': '#FFB6C1', 
    'Confusion': '#808080', 'Curiosity': '#4682B4', 'Desire': '#C71585', 
    'Disappointment': '#483D8B', 'Disapproval': '#2F4F4F', 'Disgust': '#556B2F', 
    'Embarrassment': '#DDA0DD', 'Excitement': '#FFD700', 'Fear': '#191970', 
    'Gratitude': '#00FA9A', 'Grief': '#2F2F2F', 'Joy': '#FFDF00', 
    'Love': '#FF1493', 'Nervousness': '#D2691E', 'Optimism': '#98FB98', 
    'Pride': '#8A2BE2', 'Realization': '#00CED1', 'Relief': '#20B2AA', 
    'Remorse': '#708090', 'Sadness': '#1E90FF', 'Surprise': '#FF4500'
}

# --- HELPER FUNCTIONS ---
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name)).strip()

@st.cache_data
def load_data():
    df = pd.read_csv("go_emotions_data.csv")
    
    metric_cols = ['lexical_richness', 'lexical_density'] + [col for col in df.columns if col.startswith('go_')]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['year_clean'] = pd.to_numeric(df['year'], errors='coerce')
    decades_temp = (df['year_clean'].fillna(0) // 10 * 10).astype(int).astype(str) + "s"
    df['decade'] = np.where(df['year_clean'].notna(), decades_temp, "Unknown")
    
    return df

def get_lyrics(musical, subsection, song_title):
    safe_musical = sanitize_filename(musical)
    safe_sub = sanitize_filename(subsection)
    safe_song = sanitize_filename(song_title)
    
    path = os.path.join("data", safe_musical, safe_sub, f"{safe_song}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Lyrics file not found."

# --- TERMS AND CONDITIONS TEXT ---
def render_terms_and_conditions():
    st.title("📜 Terms and Conditions")
    st.markdown("Please read these terms and conditions carefully before using the application.")
    st.markdown("---")
    
    st.subheader("1. Project Nature and Purpose")
    st.write(
        "This application is an academic, non-commercial research project developed for "
        "ENES Morelia. Its purpose is to explore computational linguistics, text mining, "
        "and sentiment/emotion classification within musical theatre repertoire."
    )
    
    st.subheader("2. Intellectual Property and Fair Use")
    st.write(
        "All musical lyrics displayed or referenced within this system are the property and "
        "copyright of their respective creators, writers, and music publishers. Lyrics are handled "
        "exclusively for non-profit educational research and quantitative analysis under Fair Use guidelines. "
        "Users are prohibited from copying, distributing, or repurposing lyrics for commercial intentions."
    )
    
    st.subheader("3. Disclaimer of Accuracy")
    st.write(
        "The lexical and emotional metrics displayed on this dashboard are generated programmatically "
        "via automated natural language processing (NLP) models (e.g., GoEmotions tokenizer). These measurements "
        "represent statistical estimates of text characteristics and may not match subjective human interpretation "
        "or artistic nuance."
    )
    
    st.subheader("4. Limitation of Liability")
    st.write(
        "This tool is provided 'as is' without warranties of any kind. The developers do not assume "
        "responsibility for temporary server downtimes, data transmission latencies, or omissions in the "
        "underlying dataset gathered from web sources."
    )
    
    st.subheader("5. Contact and Modifications")
    st.write(
        "The developer retains the authorization to update these terms or modify the application architecture "
        "at any time to preserve compliance with academic and legal boundaries."
    )

# --- MAIN APP ---
def main():
    # --- NAVIGATION VIEW ROUTING ---
    st.sidebar.header("🗺️ Navigation")
    app_mode = st.sidebar.radio("Go to:", ["📊 Analytics Dashboard", "📜 Terms & Conditions"])
    st.sidebar.markdown("---")

    if app_mode == "📜 Terms & Conditions":
        render_terms_and_conditions()
    else:
        # --- DASHBOARD HEADER ---
        st.title(PROJECT_NAME)
        st.markdown("Analyze the lexical complexity and granular emotional arcs of musical theatre using 28 emotion classes.")
        st.markdown("---")
        
        df = load_data()
        
        # --- SIDEBAR FILTERS ---
        st.sidebar.header("Stage Directions (Filters)")
        
        decades = ["All"] + sorted([d for d in df['decade'].unique() if d != "Unknown"]) + ["Unknown"]
        selected_decade = st.sidebar.selectbox("Select Decade/Era", decades)
        
        if selected_decade != "All":
            df_filtered = df[df['decade'] == selected_decade]
        else:
            df_filtered = df
            
        musicals = ["All"] + sorted(df_filtered['musical'].unique().tolist())
        selected_musical = st.sidebar.selectbox("Select Musical", musicals)
        
        if selected_musical != "All":
            df_filtered = df_filtered[df_filtered['musical'] == selected_musical]
            
        songs = ["All"] + sorted(df_filtered['song_title'].unique().tolist())
        selected_song = st.sidebar.selectbox("Select Song", songs)
        
        if selected_song != "All":
            df_filtered = df_filtered[df_filtered['song_title'] == selected_song]

        # --- KPI METRICS ---
        col1, col2, col3, col4 = st.columns(4)
        avg_richness = df_filtered['lexical_richness'].mean()
        avg_density = df_filtered['lexical_density'].mean()
        total_songs = len(df_filtered)
        unique_composers = df_filtered['composer'].nunique()
        
        col1.metric("Total Songs Selected", total_songs)
        col2.metric("Avg Lexical Richness", f"{avg_richness:.2f}")
        col3.metric("Avg Lexical Density", f"{avg_density:.2f}")
        col4.metric("Unique Composers", unique_composers)
        
        st.markdown("---")

        # --- VISUALIZATIONS: TOP SECTION ---
        v_col1, v_col2 = st.columns((3, 2))
        
        emotion_cols = [c for c in df_filtered.columns if c.startswith('go_') and c != 'go_neutral']
        
        with v_col1:
            st.subheader("Emotional Stat Build (Top 10)")
            avg_emotions = df_filtered[emotion_cols].mean().reset_index()
            avg_emotions.columns = ['Emotion', 'Score']
            
            avg_emotions['Emotion'] = avg_emotions['Emotion'].str.replace('go_', '').str.replace('_', ' ').str.title()
            
            avg_emotions = avg_emotions.sort_values(by='Score', ascending=False).head(10)
            avg_emotions = avg_emotions.sort_values(by='Emotion', ascending=True)
            avg_emotions = pd.concat([avg_emotions, avg_emotions.iloc[[0]]])
            
            marker_colors = [EMOTION_COLORS.get(emo, '#333333') for emo in avg_emotions['Emotion']]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=avg_emotions['Score'],
                theta=avg_emotions['Emotion'],
                fill='toself',
                fillcolor='rgba(113, 108, 176, 0.2)',
                opacity=0.9,
                mode='lines+markers',
                line=dict(color='rgba(113, 108, 176, 0.6)', width=2),
                marker=dict(
                    color=marker_colors,
                    size=14,
                    line=dict(color='white', width=1.5)
                ),
                hoverinfo='text',
                text=[f"{emo}: {score:.3f}" for emo, score in zip(avg_emotions['Emotion'], avg_emotions['Score'])]
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True)
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with v_col2:
            st.subheader("Lexical Metrics")
            lex_data = pd.DataFrame({
                "Metric": ["Richness", "Density"],
                "Value": [avg_richness, avg_density]
            })
            fig_bar = px.bar(
                lex_data, 
                x="Metric", 
                y="Value",
                color="Metric",
                color_discrete_sequence=["#79e0cf", "#e07ab3"]
            )
            fig_bar.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 1]),
                yaxis_title="Score (0 to 1)",
                xaxis_title=None
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # --- NARRATIVE ARC (LINE CHART) ---
        if selected_musical != "All" and selected_song == "All":
            st.subheader(f"The Narrative Arc of {selected_musical}")
            st.markdown("Track the emotional journey sequentially from the opening number to the finale.")
            
            top_emotions_musical = df_filtered[emotion_cols].mean().sort_values(ascending=False).head(3).index.tolist()
            clean_top = [e.replace('go_', '').replace('_', ' ').title() for e in top_emotions_musical]
            clean_all = [e.replace('go_', '').replace('_', ' ').title() for e in emotion_cols]
            
            selected_arc_emotions = st.multiselect(
                "Select emotions to track:", 
                options=sorted(clean_all), 
                default=clean_top
            )
            
            if selected_arc_emotions:
                target_cols = [f"go_{e.lower().replace(' ', '_')}" for e in selected_arc_emotions]
                arc_df = df_filtered[['song_title'] + target_cols].copy()
                
                arc_melt = arc_df.melt(id_vars=['song_title'], value_vars=target_cols, var_name='Emotion', value_name='Score')
                arc_melt['Emotion'] = arc_melt['Emotion'].str.replace('go_', '').str.replace('_', ' ').str.title()
                
                line_colors = {emo: EMOTION_COLORS.get(emo, '#333333') for emo in selected_arc_emotions}
                
                fig_arc = px.line(
                    arc_melt, 
                    x='song_title', 
                    y='Score', 
                    color='Emotion', 
                    markers=True,
                    color_discrete_map=line_colors
                )
                fig_arc.update_layout(
                    xaxis_title="Song (Sequential Order)", 
                    yaxis_title="Emotion Score",
                    xaxis={'tickangle': -45},
                    margin=dict(b=100)
                )
                st.plotly_chart(fig_arc, use_container_width=True)
                st.markdown("---")

        # --- EMOTION CO-OCCURRENCE HEATMAP ---
        if selected_song == "All":
            with st.expander("🔥 Emotion Co-Occurrence Heatmap", expanded=False):
                st.markdown("Discover which emotions frequently appear together. A score close to **1** means they spike together, while **-1** means they rarely mix.")
                
                corr_matrix = df_filtered[emotion_cols].corr()
                clean_names = [c.replace('go_', '').replace('_', ' ').title() for c in corr_matrix.columns]
                corr_matrix.columns = clean_names
                corr_matrix.index = clean_names
                
                fig_heat = px.imshow(
                    corr_matrix, 
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, 
                    zmax=1,
                    aspect="auto"
                )
                fig_heat.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_heat, use_container_width=True)

        # --- TOP WORDS, METADATA & LYRICS ---
        if selected_musical != "All":
            meta_row = df_filtered.iloc[0]
            
            if selected_song != "All":
                st.subheader(f"Song: {selected_song}")
                st.markdown(f"**Musical:** {selected_musical} | **Act/Subsection:** {meta_row['subsection']} | **Year:** {meta_row['year']} | **Composer:** {meta_row['composer']} | **Lyricist:** {meta_row['lyricist']}")
            else:
                st.subheader(f"Musical: {selected_musical}")
                st.markdown(f"**Year:** {meta_row['year']} | **Composer:** {meta_row['composer']} | **Lyricist:** {meta_row['lyricist']}")
            
            st.write("")
            all_words = df_filtered['most_used_words'].dropna().str.split(', ').sum()
            if isinstance(all_words, list):
                top_words = pd.Series(all_words).value_counts().head(10)
                st.write("**Top Words Used:**")
                words_display = " • ".join([f"**{word}** ({count})" for word, count in top_words.items()])
                st.info(words_display)
                
            st.write("")
            
            if selected_song != "All":
                song_row = df_filtered.iloc[0]
                with st.expander(f"📖 Read Lyrics: {selected_song}", expanded=True):
                    lyrics = get_lyrics(song_row['musical'], song_row['subsection'], song_row['song_title'])
                    st.text(lyrics)
            else:
                st.dataframe(df_filtered[['subsection', 'song_title', 'lexical_richness', 'lexical_density']], use_container_width=True)

    # --- GLOBAL FOOTER ---
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #808080; padding: 10px;'>{FOOTER_MESSAGE}</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
