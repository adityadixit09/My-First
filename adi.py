import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Page config - Netflix style
st.set_page_config(
    page_title="Netflix Analytics Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;600;700&display=swap');
    
    .main { background: linear-gradient(135deg, #141414 0%, #000000 50%, #141414 100%); }
    .stApp { background: linear-gradient(135deg, #141414 0%, #000000 50%, #141414 100%); }
    
    .netflix-header {
        background: linear-gradient(90deg, #E50914 0%, #B20710 100%);
        padding: 1rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3);
    }
    
    .kpi-metric {
        background: rgba(31, 31, 31, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .kpi-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(229, 9, 20, 0.4);
        border-color: #E50914;
    }
    
    .chart-container {
        background: rgba(31, 31, 31, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        border: 1px solid rgba(229, 9, 20, 0.2);
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .sidebar .sidebar-content {
        background: rgba(20, 20, 20, 0.95);
        backdrop-filter: blur(20px);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #E50914, #B20710);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(229, 9, 20, 0.6);
    }
    
    h1, h2, h3 { color: #FFFFFF; font-family: 'Netflix Sans', sans-serif; }
    .stMetric { color: #FFFFFF; }
    </style>
""", unsafe_allow_html=True)

# Load and clean data
@st.cache_data
def load_data():
    # Create sample Netflix dataset (replace with your CSV)
    np.random.seed(42)
    n_titles = 8800
    
    data = {
        'show_id': range(1, n_titles + 1),
        'type': np.random.choice(['Movie', 'TV Show'], n_titles, p=[0.7, 0.3]),
        'title': [f"Title {i} - {'Movie' if np.random.rand()>0.3 else 'Series'}" for i in range(1, n_titles + 1)],
        'director': np.random.choice(['Michael Bay', 'Christopher Nolan', 'Greta Gerwig', 'Denis Villeneuve', 'Unknown'], n_titles),
        'cast': np.random.choice(['Tom Cruise', 'Leonardo DiCaprio', 'Margot Robbie', 'Ryan Gosling', 'Multiple'], n_titles),
        'country': np.random.choice(['United States', 'India', 'UK', 'Japan', 'South Korea', 'Canada'], n_titles, p=[0.45, 0.25, 0.1, 0.08, 0.07, 0.05]),
        'date_added': pd.date_range('2010-01-01', periods=n_titles, freq='D').strftime('%Y-%m-%d'),
        'release_year': np.random.randint(1990, 2024, n_titles),
        'rating': np.random.choice(['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'PG', 'G'], n_titles),
        'duration': np.random.choice([90, 120, 150, 25, 45, 60], n_titles),
        'listed_in': np.random.choice([
            'Dramas', 'Comedies', 'Documentaries', 'Action & Adventure', 
            'Children & Family Movies', 'Thrillers', 'Horror Movies', 'Anime'
        ], n_titles),
        'description': ['Amazing story about life and adventure.' for _ in range(n_titles)]
    }
    
    df = pd.DataFrame(data)
    return df

df = load_data()

# Recommendation Engine
@st.cache_data
def get_recommendations(title, top_n=10):
    # Simple content-based filtering
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['listed_in'] + ' ' + df['country'])
    
    idx = df[df['title'].str.contains(title, case=False, na=False)].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'type', 'listed_in', 'rating', 'release_year']]

# Header
st.markdown("""
    <div class="netflix-header">
        <h1 style='margin: 0; font-size: 2.5rem; font-weight: 700;'>
            🎬 Netflix Analytics Pro
        </h1>
        <p style='margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 1.1rem;'>
            Explore 8,800+ titles • Advanced Insights • Smart Recommendations
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Filters
st.sidebar.title("🎛️ Control Panel")
page = st.sidebar.selectbox(
    "Navigate",
    ["📊 Overview", "📈 Content Trends", "🌍 Global Reach", "⏱️ Duration Analysis", 
     "🔍 Search & Discovery", "🤖 Recommendations"]
)

# Global Filters
st.sidebar.subheader("🔧 Filters")
type_filter = st.sidebar.multiselect("Type", ['Movie', 'TV Show'], default=['Movie', 'TV Show'])
genre_filter = st.sidebar.multiselect("Genre", df['listed_in'].unique(), default=df['listed_in'].value_counts().head(5).index.tolist())
country_filter = st.sidebar.multiselect("Country", df['country'].unique(), default=['United States', 'India'])
year_range = st.sidebar.slider("Release Year", 1990, 2023, (2010, 2023))

# Apply filters
filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['listed_in'].isin(genre_filter)) &
    (df['country'].isin(country_filter)) &
    (df['release_year'].between(year_range[0], year_range[1]))
]

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="kpi-metric">
            <h3 style='font-size: 2.5rem; margin: 0; color: #E50914;'>{}</h3>
            <p style='color: #b3b3b3; margin: 0.5rem 0 0 0;'>Total Titles</p>
        </div>
    """.format(len(filtered_df)), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="kpi-metric">
            <h3 style='font-size: 2.5rem; margin: 0; color: #F4B400;'>{}</h3>
            <p style='color: #b3b3b3; margin: 0.5rem 0 0 0;'>Movies</p>
        </div>
    """.format(len(filtered_df[filtered_df['type']=='Movie'])), unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="kpi-metric">
            <h3 style='font-size: 2.5rem; margin: 0; color: #00A3E0;'>{}</h3>
            <p style='color: #b3b3b3; margin: 0.5rem 0 0 0;'>TV Shows</p>
        </div>
    """.format(len(filtered_df[filtered_df['type']=='TV Show'])), unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="kpi-metric">
            <h3 style='font-size: 2.5rem; margin: 0; color: #7C2D12;'>{}</h3>
            <p style='color: #b3b3b3; margin: 0.5rem 0 0 0;'>Countries</p>
        </div>
    """.format(filtered_df['country'].nunique()), unsafe_allow_html=True)

st.markdown("---")

# Main Content by Page
if page == "📊 Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Content Type Distribution
        type_counts = filtered_df['type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values, 
            names=type_counts.index,
            color_discrete_sequence=['#E50914', '#F4B400'],
            hole=0.6
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Top Genres
        genre_counts = filtered_df['listed_in'].value_counts().head(8)
        fig_bar = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            color=genre_counts.values,
            color_continuous_scale='reds'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "📈 Content Trends":
    st.header("📈 Content Trends Over Time")
    
    yearly_trends = filtered_df.groupby(['release_year', 'type']).size().reset_index(name='count')
    fig_line = px.line(
        yearly_trends, 
        x='release_year', 
        y='count',
        color='type',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#F4B400'},
        title="Content Growth by Year"
    )
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_line, use_container_width=True)

elif page == "🌍 Global Reach":
    st.header("🌍 Global Content Distribution")
    
    country_counts = filtered_df['country'].value_counts().head(10)
    fig_choro = px.choropleth(
        country_counts.reset_index(),
        locations="index",
        locationmode="country names",
        color="country",
        hover_name="index",
        color_continuous_scale="reds",
        title="Content by Country"
    )
    fig_choro.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_choro, use_container_width=True)

elif page == "⏱️ Duration Analysis":
    st.header("⏱️ Duration Distribution")
    
    if filtered_df['type'].eq('Movie').any():
        movie_duration = filtered_df[filtered_df['type']=='Movie']['duration']
        fig_hist = px.histogram(
            movie_duration, 
            nbins=30,
            title="Movie Duration Distribution",
            color_discrete_sequence=['#E50914']
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🔍 Search & Discovery":
    st.header("🔍 Search & Discovery")
    
    search_query = st.text_input("🔎 Search titles, directors, actors...", key="search")
    
    if search_query:
        search_results = filtered_df[
            filtered_df['title'].str.contains(search_query, case=False, na=False) |
            filtered_df['director'].str.contains(search_query, case=False, na=False) |
            filtered_df['cast'].str.contains(search_query, case=False, na=False)
        ]
        
        if not search_results.empty:
            st.success(f"✅ Found {len(search_results)} results")
            st.dataframe(search_results[['title', 'type', 'director', 'cast', 'rating', 'release_year']], use_container_width=True)
        else:
            st.warning("❌ No results found")

elif page == "🤖 Recommendations":
    st.header("🤖 Smart Recommendations")
    
    st.info("🔮 Enter a movie/series title to get personalized recommendations!")
    
    rec_input = st.text_input("Enter title for recommendations:", placeholder="e.g., Title 1234")
    
    if st.button("🎯 Get Recommendations", key="recommend") and rec_input:
        with st.spinner("Computing recommendations..."):
            recommendations = get_recommendations(rec_input, top_n=10)
            
        if not recommendations.empty:
            st.success(f"🎉 Top 10 recommendations for '{rec_input}':")
            
            # Recommendation cards
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**{row['title']}**")
                with col2:
                    st.caption(row['type'])
                with col3:
                    st.caption(row['listed_in'])
                with col4:
                    st.caption(row['rating'])
                st.markdown("---")
        else:
            st.error("No recommendations found!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #b3b3b3;'>
        <h3>🎬 Built with Streamlit + Plotly + Pandas</h3>
        <p>Netflix Dataset Analysis | Deployed on Streamlit Cloud</p>
    </div>
""", unsafe_allow_html=True)