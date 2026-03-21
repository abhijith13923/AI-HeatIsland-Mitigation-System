import re

with open('c:\\Users\\uditt\\Desktop\\EPICS\\app.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_css = '''    <style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #121212;
        color: #e0e0e0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #bb86fc, #03dac6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3.2rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Card Styling */
    .metric-card {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 16px;
        padding: 25px 15px;
        border: 1px solid #333333;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7);
        border-color: #bb86fc;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #03dac6;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #aaaaaa;
        margin-top: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: #bb86fc;
        color: #000000;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        transition: all 0.2s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: #3700b3;
        color: #ffffff;
    }
    
    /* Result Box */
    .result-box {
        background: #1e1e1e;
        border-radius: 20px;
        padding: 40px 20px;
        margin-top: 30px;
        border: 1px solid #333333;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .severity-text {
        font-size: 4rem;
        font-weight: 900;
        margin: 15px 0;
        letter-spacing: 2px;
    }
    
    /* Severity Colors */
    .sev-0 { color: #03dac6 !important; }
    .sev-1 { color: #fbbc04 !important; }
    .sev-2 { color: #ff9100 !important; }
    .sev-3 { color: #cf6679 !important; }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background: #181818;
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: #bb86fc;
    }
    
    /* Map Container */
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #333333;
    }
    </style>'''

new_text = re.sub(r'<style>.*?</style>', new_css, text, flags=re.DOTALL)

with open('c:\\Users\\uditt\\Desktop\\EPICS\\app.py', 'w', encoding='utf-8') as f:
    f.write(new_text)
