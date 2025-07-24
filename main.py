from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="🎵 Song Recommendation App",
    page_icon="🎵",
    layout="wide"
)

# Initialize the LLM
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        groq_api_key=api_key,
        model="llama3-70b-8192",
        temperature=1
    )

llm = initialize_llm()

# Create prompt template
prompt_template = PromptTemplate(
    input_variables=['mood', 'language'],
    template="This is my {mood} right now suggest me a list of top 10 songs with this {mood} as its genre within this {language} language"
)

# Streamlit App UI
st.title("🎵 Personalized Song Recommendation App")
st.markdown("Get song recommendations based on your current mood and preferred language!")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎭 Your Current Mood")
    mood_options = [
        "romantic", "happy", "sad", "energetic", "relaxed", 
        "melancholic", "upbeat", "nostalgic", "dreamy", "motivational"
    ]
    selected_mood = st.selectbox("Choose your mood:", mood_options)
    
    # Option to add custom mood
    custom_mood = st.text_input("Or enter a custom mood:")
    if custom_mood:
        selected_mood = custom_mood

with col2:
    st.subheader("🌍 Preferred Language")
    language_options = [
        "English", "Hindi", "Tamil", "Telugu", "Malayalam", 
        "Kannada", "Bengali", "Punjabi", "Marathi", "Gujarati",
        "Spanish", "French", "Korean", "Japanese"
    ]
    selected_language = st.selectbox("Choose language:", language_options)
    
    # Option to add custom language
    custom_language = st.text_input("Or enter a custom language:")
    if custom_language:
        selected_language = custom_language

# Generate recommendations button
st.markdown("---")
if st.button("🎵 Get Song Recommendations", type="primary", use_container_width=True):
    if not api_key:
        st.error("❌ GROQ API key not found! Please add it to your .env file.")
        st.info("Create a .env file with: GROQ_API_KEY=your_api_key_here")
    else:
        # Show loading spinner
        with st.spinner(f"Finding the perfect {selected_mood} songs in {selected_language}..."):
            try:
                # Format the prompt
                formatted_prompt = prompt_template.format(
                    mood=selected_mood, 
                    language=selected_language
                )
                
                # Get response from LLM
                response = llm.invoke(formatted_prompt)
                
                # Display results
                st.success("✅ Here are your personalized song recommendations!")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["🎵 Song List", "📝 Full Response"])
                
                with tab1:
                    st.markdown(f"### {selected_mood.title()} Songs in {selected_language}")
                    # Format the response nicely
                    songs = response.content.split('\n')
                    for song in songs:
                        if song.strip() and any(char.isdigit() for char in song):
                            st.markdown(f"• {song.strip()}")
                
                with tab2:
                    st.markdown("### Complete AI Response:")
                    st.text_area("", response.content, height=400, disabled=True)
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please check your API key and internet connection.")

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses AI to recommend songs based on:
    - **Your current mood** 🎭
    - **Preferred language** 🌍
    
    **How to use:**
    1. Select your mood
    2. Choose your language
    3. Click 'Get Recommendations'
    4. Enjoy your personalized playlist! 🎵
    """)
    
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("• LangChain 🦜")
    st.markdown("• Groq AI ⚡")
    st.markdown("• Streamlit 🚀")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit and LangChain</p>", 
    unsafe_allow_html=True
)

