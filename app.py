import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Netflix Analytics Pro 🎬",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Netflix CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
* { font-family: 'Inter', sans-serif; }

.main { 
    background: linear-gradient(135deg, #141414 0%, #000000 50%, #141414 100%);
    padding-top: 2rem;
}

.stApp { 
    background: linear-gradient(135deg, #141414 0%, #000000 50%, #141414 100%);
}

.metric-container {
    background: rgba(31, 31, 31, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(229, 9, 20, 0.2) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}

.metric-container:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 20px 40px rgba(229, 9, 20, 0.4) !important;
    border-color: #E50914 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #E50914, #B20710) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(229, 9, 20, 0.6) !important;
}

h1, h2, h3 { color: #FFFFFF !important; }
.stMetric > label { color: #b3b3b3 !important; }
.stMetric > div > div { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Generate realistic Netflix dataset"""
    np.random.seed(42)
    n = 8800
    
    genres = ['Dramas', 'Comedies', 'Documentaries', 'Action & Adventure', 
              'Children & Family Movies', 'Thrillers', 'Horror Movies']
    
    countries = ['United States', 'India', 'United Kingdom', 'Japan', 
                'South Korea', 'Canada', 'France', 'Germany']
    
    ratings = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'PG', 'G']
    
    data = {
        'show_id': [f'nf{i:04d}' for i in range(1, n+1)],
        'type': np.random.choice(['Movie', 'TV Show'], n, p=[0.7, 0.3]),
        'title': [f"Title {i} - {'Movie' if np.random.rand()>0.3 else 'Series'}" for i in range(1, n+1)],
        'director': np.random.choice(['Michael Bay', 'Christopher Nolan', 'Greta Gerwig', 
                                    'Denis Villeneuve', 'Unknown'], n),
        'cast': np.random.choice(['Tom Cruise', 'Leonardo DiCaprio', 'Margot Robbie', 
                                'Ryan Gosling', 'Multiple Cast'], n),
        'country': np.random.choice(countries, n, p=[0.45, 0.25, 0.1, 0.08, 0.07, 0.03, 0.01, 0.01]),
        'release_year': np.random.randint(1990, 2024, n),
        'rating': np.random.choice(ratings, n),
        'duration': np.random.choice([90, 120, 150, 25, 45, 60], n),
        'listed_in': np.random.choice(genres, n),
        'description': ['Amazing story about life, love, and adventure.' for _ in range(n)]
    }
    
    return pd.DataFrame(data)

# Load data
df = load_data()

# Recommendation function
@st.cache_data
def get_recommendations(title, df, top_n=10):
    """Content-based recommendation using TF-IDF"""
    try:
        # Create features
        df_features = df.copy()
        df_features['features'] = df_features['listed_in'] + ' ' + df_features['country'] + ' ' + df_features['rating']
        
        # TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df_features['features'])
        
        # Find title index
        idx = df_features[df_features['title'].str.contains(title, case=False, na=False)].index[0]
        
        # Cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        return df_features.iloc[movie_indices][['title', 'type', 'listed_in', 'rating', 'release_year']].reset_index(drop=True)
    except:
        return pd.DataFrame()

# Header
st.markdown("""
<div style='
    background: linear-gradient(90deg, #E50914 0%, #B20710 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(229, 9, 20, 0.4);
    text-align: center;
'>
    <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; color: white;'>🎬 Netflix Analytics Pro</h1>
    <p style='margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 1.2rem;'>
        Advanced Insights • 8,800+ Titles • Smart Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    page = st.selectbox("📂 Navigation", [
        "📊 Overview", "📈 Content Trends", "🌍 Global Reach", 
        "⏱️ Duration Analysis", "🔍 Search", "🤖 Recommendations"
    ])
    
    st.subheader("🔧 Filters")
    type_filter = st.multiselect("Type", ['Movie', 'TV Show'], default=['Movie', 'TV Show'])
    genre_filter = st.multiselect("Genre", sorted(df['listed_in'].unique()), key="genres")
    country_filter = st.multiselect("Country", sorted(df['country'].unique()), key="countries")
    year_range = st.slider("Release Year", 1990, 2023, (2010, 2023))

# Filter data
filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['listed_in'].isin(genre_filter)) &
    (df['country'].isin(country_filter)) &
    (df['release_year'].between(*year_range))
]

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Titles", len(filtered_df), len(df))
with col2:
    st.metric("Movies", len(filtered_df[filtered_df['type']=='Movie']), len(df[df['type']=='Movie']))
with col3:
    st.metric("TV Shows", len(filtered_df[filtered_df['type']=='TV Show']), len(df[df['type']=='TV Show']))
with col4:
    st.metric("Countries", filtered_df['country'].nunique(), df['country'].nunique())

st.markdown("---")

# Pages
if page == "📊 Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        type_counts = filtered_df['type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values, 
            names=type_counts.index,
            color_discrete_sequence=['#E50914', '#F4B400'],
            hole=0.6,
            title="Content Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Top genres
        genre_counts = filtered_df['listed_in'].value_counts().head(8)
        fig_bar = px.bar(
            genre_counts, 
            orientation='h',
            color=genre_counts.values,
            color_continuous_scale='reds',
            title="Top Genres"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "📈 Content Trends":
    st.header("📈 Content Trends Over Time")
    
    yearly_data = filtered_df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
    fig_line = px.line(
        yearly_data, 
        title="Releases by Year",
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#F4B400'}
    )
    st.plotly_chart(fig_line, use_container_width=True)

elif page == "🌍 Global Reach":
    st.header("🌍 Global Content Distribution")
    
    country_data = filtered_df['country'].value_counts().head(15).reset_index()
    country_data.columns = ['country', 'count']
    
    fig_map = px.choropleth(
        country_data, 
        locations="country",
        locationmode="country names",
        color="count",
        hover_name="country",
        color_continuous_scale="reds",
        title="Content Production by Country"
    )
    st.plotly_chart(fig_map, use_container_width=True)

elif page == "⏱️ Duration Analysis":
    st.header("⏱️ Duration Analysis")
    
    if 'Movie' in filtered_df['type'].values:
        movie_data = filtered_df[filtered_df['type']=='Movie']['duration']
        fig_hist = px.histogram(
            movie_data, 
            nbins=30,
            title="Movie Duration Distribution (minutes)",
            color_discrete_sequence=['#E50914']
        )
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🔍 Search":
    st.header("🔍 Search & Discovery")
    
    search_query = st.text_input("Search titles, directors, actors...", value="")
    
    if search_query:
        results = filtered_df[
            filtered_df['title'].str.contains(search_query, case=False, na=False) |
            filtered_df['director'].str.contains(search_query, case=False, na=False) |
            filtered_df['cast'].str.contains(search_query, case=False, na=False)
        ]
        
        if len(results) > 0:
            st.success(f"✅ Found {len(results)} results")
            st.dataframe(
                results[['title', 'type', 'director', 'cast', 'rating', 'release_year']],
                use_container_width=True
            )
        else:
            st.warning("❌ No results found")

elif page == "🤖 Recommendations":
    st.header("🤖 AI Recommendations")
    
    st.info("🔮 Enter a title to get personalized recommendations based on genre, country & rating")
    
    rec_input = st.text_input("Enter title:", placeholder="e.g., Title 1001")
    
    if st.button("🎯 Generate Recommendations") and rec_input:
        with st.spinner("🤖 Computing recommendations..."):
            recommendations = get_recommendations(rec_input, filtered_df)
            
        if not recommendations.empty:
            st.success(f"🎉 Top recommendations for '{rec_input}':")
            
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1.5, 1])
                    col1.markdown(f"**{row['title']}**")
                    col2.metric("Type", row['type'])
                    col3.metric("Genre", row['listed_in'])
                    col4.metric("Rating", row['rating'])
                    st.markdown("---")
        else:
            st.error("No recommendations found!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem; color: #b3b3b3;'>
        <h3>🎬 Netflix Analytics Dashboard</h3>
        <p>Built with Streamlit + Plotly + Pandas | Deployed on Streamlit Cloud 🚀</p>
    </div>
    """, 
    unsafe_allow_html=True
)