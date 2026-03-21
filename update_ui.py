import re

with open('c:\\Users\\uditt\\Desktop\\EPICS\\app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Header
header_pattern = r'# Main Content - Enhanced.*?st\.markdown\(\'<div class="custom-divider"></div>\', unsafe_allow_html=True\)'
new_header = '''# Main Content - Enhanced
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="background: linear-gradient(90deg, #4ade80, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 800; margin-bottom: 0;">🌿 Urban Heat Island Severity</h1>
    <p style="color: #a1a1aa; font-size: 1.2rem; font-weight: 400; margin-top: 5px;">AI-Powered Environmental Intelligence for Sustainable Cities</p>
    <div style="display: inline-block; background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.5); color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; margin-top: 10px;">
        🔴 LIVE ANALYSIS
    </div>
</div>
""", unsafe_allow_html=True)'''
text = re.sub(header_pattern, new_header, text, flags=re.DOTALL)

# Replace CSS
css_pattern = r'<style>.*?</style>'
new_css = '''<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #09090b;
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card Enhancements */
    .metric-card, .result-box, .map-container {
        background: #18181b;
        border-radius: 16px;
        border: 1px solid #27272a;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -2px rgba(0, 0, 0, 0.5);
    }
    
    .metric-card {
        padding: 24px;
        transition: all 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -4px rgba(0, 0, 0, 0.5);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4ade80;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a1a1aa;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result Box */
    .result-box {
        padding: 40px 20px;
        margin-top: 30px;
        text-align: center;
        background: linear-gradient(145deg, #18181b, #09090b);
    }
    
    .severity-text {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 15px 0;
        letter-spacing: 2px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    /* Severity Colors */
    .sev-0 { color: #4ade80 !important; }
    .sev-1 { color: #fbbf24 !important; }
    .sev-2 { color: #f97316 !important; }
    .sev-3 { color: #ef4444 !important; }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #4ade80) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #22c55e) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background: #09090b;
        border-right: 1px solid #27272a;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #4ade80);
    }
    </style>'''
text = re.sub(css_pattern, new_css, text, flags=re.DOTALL)

# Replace hardcoded box colors
text = re.sub(r'background: rgba\(20, 60, 45, 0\.3\); border-radius: 30px; padding: 25px; border: 1px solid rgba\(105,240,174,0\.1\);',
              r'background: #18181b; border-radius: 16px; padding: 25px; border: 1px solid #27272a;', text)
text = re.sub(r'background: rgba\(20, 60, 45, 0\.3\); border-radius: 30px; padding: 20px; border: 1px solid rgba\(105,240,174,0\.1\);',
              r'background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;', text)
text = re.sub(r'background: rgba\(20, 60, 45, 0\.2\); border-radius: 20px; padding: 20px;',
              r'background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;', text)
text = re.sub(r'background: rgba\(20, 60, 45, 0\.3\); border-radius: 20px; padding: 25px;',
              r'background: #18181b; border-radius: 16px; padding: 25px; border: 1px solid #27272a;', text)

# Sidebar gradient border replace
text = re.sub(r'background: linear-gradient\(145deg, #0a3a2a, #052018\); border-radius: 20px; padding: 20px; border: 1px solid rgba\(105,240,174,0\.2\);',
              r'background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;', text)

# Stats boxes
text = re.sub(r'background: rgba\(105,240,174,0\.1\); border-radius: 15px; padding: 15px; text-align: center;',
              r'background: #27272a; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #3f3f46;', text)

# Welcome box
text = re.sub(r'background: linear-gradient\(145deg, rgba\(20,80,60,0\.2\), rgba\(10,50,40,0\.2\)\);.*?border: 1px solid rgba\(105,240,174,0\.1\);',
              r'background: #18181b; border-radius: 20px; padding: 40px; text-align: center; margin-top: 40px; border: 1px solid #27272a;', text, flags=re.DOTALL)

# Gem AI box
text = re.sub(r'background: rgba\(30,30,30,0\.8\); padding: 25px; border-radius: 15px; border: 1px solid #444;',
              r'background: #18181b; padding: 30px; border-radius: 16px; border: 1px solid #3b82f6; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);', text)

with open('c:\\Users\\uditt\\Desktop\\EPICS\\app.py', 'w', encoding='utf-8') as f:
    f.write(text)
