import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np # Import numpy for creating synthetic data

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Define the prediction labels
LABEL_MAP = {0: "REAL NEWS ✅", 1: "FAKE NEWS 🛑"}
TARGET_NAMES = ["REAL (0)", "FAKE (1)"]

# --- REMOVED HARDCODED CREDENTIALS ---

# NOTE: For a real project, you would load a massive dataset from a CSV file.
# We are using an embedded dataset for this single-file demonstration.
EMBEDDED_DATA = {
    'text': [
        # --- FAKE NEWS Examples (Label 1) ---
        "URGENT: World to end tomorrow due to secret government experiment!",
        "Click here now to instantly win a million dollars with this simple trick!",
        "President secretly dissolved the Congress and declared martial law.",
        "Shocking video proves aliens landed in New York last night.",
        "Unbelievable trick to instantly lose 50 pounds without any effort.",
        "A celebrity died and the news channels are hiding it from the public.",
        "Free Tesla for everyone who shares this post!",
        "New study proves coffee cures all types of cancer, doctors baffled.",
        # --- REAL NEWS Examples (Label 0) ---
        "The Senate passed a spending bill by a vote of 51 to 49 yesterday.",
        "Researchers published new findings on quantum computing in Nature journal.",
        "Local city council approves budget for new road construction in the northern district.",
        "Stocks rose sharply today following positive Q3 economic reports.",
        "The meteorological department forecasts heavy rainfall across the southern region.",
        "Interview with the winner of the Nobel Prize for Literature 2024.",
        "NASA successfully launched a new observational satellite into orbit.",
        "New retail store opened downtown, offering discounts to early customers.",
    ],
    'label': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
}

# BACKGROUND IMAGE URL
BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1549490349-0e9ad1a92b2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80"

@st.cache_resource
def train_model():
    """
    Loads data, performs preprocessing (TF-IDF), and trains the best model.
    Returns the vectorizer, model, and the test results for the analytics page.
    """
    
    # Load and prepare data
    df = pd.DataFrame(EMBEDDED_DATA)
    X = df['text']
    y = df['label']

    # Text Preprocessing Function
    def preprocess_text(text):
        # Remove special characters, numbers, and multiple spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        text = text.strip()
        return text

    X_processed = X.apply(preprocess_text)

    # Split data (necessary for the vectorizer fit and model evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # --- Feature Extraction: TF-IDF Vectorization ---
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # --- Model Training: Passive Aggressive Classifier ---
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model for the report
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    
    # Return everything needed for the app and the analytics page
    return tfidf_vectorizer, model, y_test, y_pred


# --- 2. LOGIN AND AUTHENTICATION FUNCTIONS ---

def login_page():
    """Displays the login form."""
    st.set_page_config(
        page_title="Login - Hoax Detector",
        layout="centered"
    )
    def get_base64_image(image_path):
        # NOTE: This function assumes 'Bg.png' exists. Replace this with a working path or URL if not
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            # Fallback if the file is not found
            st.error(f"Background image '{image_path}' not found. Please place it in the same directory.")
            return ""

# IMPORTANT: Ensure Bg.png is in the same directory as your script
    image_base64 = get_base64_image("c.avif")  # Replace with your image file

    
    # Inject custom CSS for background image AND hiding header elements
    st.markdown(
        f"""
        <style>
        /* --- START OF HEADER HIDING CSS --- */
        header {{
            visibility: hidden;
            display: none !important;
        }}
        .main {{
            padding-top: 0;
            padding-left: 0;
            padding-right: 0;
            padding-bottom: 0;
        }}
        button[title="View source code"] {{ 
            display: none !important;
        }}
        .stDeployButton {{
            display: none !important;
        }}
        /* --- END OF HEADER HIDING CSS --- */
        
        .stApp {{
            background-image: url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stForm {{
            padding: 2rem;
            border: 1px solid #e6e9ef;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 300px;
            margin: 0 auto;
            background-color: rgba(138, 138, 175, 0.55); /* Slightly transparent white for readability */
        }}
        .stText, .stMarkdown, .stSubheader, .stLabel  {{
            color: #000000; /* Darker text for contrast on light background */
            text-shadow: 1px 1px 2px rgba(255,255,255,0.7); /* Subtle shadow for text on text */
            font-weight: 600;
        }}

        .stTitle {{
            color: #1a1a1a;
            font-weight: 1000;
            font-size: 3rem;
            text-align: center; 
            letter-spacing: 0.5px;
        }}

       </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="stTitle">Welcome to Hoax Detector</div>', unsafe_allow_html=True)

    
    
    with st.container():
        st.write("") # Spacer
        with st.form("login_form"):
            st.markdown("Please enter your credentials to access the Fake News Detector.")
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary")

            if submitted:
                # --- UPDATED LOGIN LOGIC: Allow access if both fields are not empty ---
                if username and password: 
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}! Redirecting...")
                    st.rerun()
                else:
                    st.error("Username and Password cannot be empty.")

# --- 3. MAIN APPLICATION FUNCTIONALITY ---

def detector_page(tfidf_vectorizer, model):
    """Main page for text analysis."""
    st.title("📰 Real or Hoax? AI-Powered News Detector")
    st.markdown("---")

    st.subheader("Paste your article or headline here:")

    # Input text area for the user
    user_input = st.text_area(
        "Paste Text",
        "Scientists discover a new form of water on Mars that grants immortality.",
        height=200,
        label_visibility="collapsed"
    )

    # Prediction button
    if st.button("Analyze Credibility", type="primary"):
        if not user_input.strip():
            st.error("Please enter some text to analyze.")
        else:
            # --- 4. PREDICTION LOGIC ---
            
            with st.spinner('Analyzing text structure and linguistic patterns...'):
                # Preprocess the input text (must match training preprocessing)
                processed_input = re.sub(r'[^a-zA-Z\s]', '', user_input, re.I|re.A).lower().strip()

                # Vectorize the input using the *fitted* TF-IDF vectorizer
                input_vectorized = tfidf_vectorizer.transform([processed_input])

                # Predict the class (0 or 1)
                prediction_label_index = model.predict(input_vectorized)[0]
                
                final_label = LABEL_MAP[prediction_label_index]
                
                # --- Display Results ---
                st.markdown("## Prediction Result")
                
                if prediction_label_index == 1:
                    st.error(f"**Classification:** {final_label}")
                    st.markdown("🚨 **Caution:** The model detected linguistic features common in fabricated or misleading articles. Further human verification is highly recommended.")
                else:
                    st.success(f"**Classification:** {final_label}")
                    st.markdown("✅ **Assessment:** The text's linguistic structure aligns with patterns typically found in authentic news sources.")

def analytics_page(y_test, y_pred):
    """Page to display model performance metrics and visualizations."""
    
    st.title("📈 Hoax Detector Analytics Dashboard")
    st.markdown("---")
    
    # --- SIMULATED REAL-TIME DASHBOARD (Based on User's Image) ---
    
    
    # Simulated/Hardcoded Metrics for the top cards
    TOTAL_USERS = 1203
    ACTIVE_USERS = 862
    HOAXES_DETECTED = 308
    
    # Create the three metric columns
    col1, col2, col3 = st.columns(3)
    
    # Inject custom CSS for the metric cards to look more like the image
    st.markdown(
        """
        <style>
        /* Styling for the Streamlit metric box elements */
        .css-1y4c5n1 {{
            background-color: #5d3f6a; /* Darker background color */
            padding: 15px 15px;
            border-radius: 10px;
            color: #ffffff; /* White text */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            text-align: center;
        }}
        .css-1y4c5n1 label, .css-1y4c5n1 div[data-testid="stMetricLabel"] {{
            color: #f1f1f1 !important; /* Lighter text for the label */
        }}
        .css-1y4c5n1 div[data-testid="stMetricValue"] {{
            color: #ffffff !important; /* White text for the value */
            font-size: 2.5em !important;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    with col1:
        st.metric(label="Total Users", value=TOTAL_USERS, delta=55, delta_color="normal")
    
    with col2:
        st.metric(label="Active Users", value=ACTIVE_USERS, delta=20, delta_color="normal")
        
    with col3:
        # Assuming last period had 290 hoaxes detected
        st.metric(label="Hoaxes Detected", value=HOAXES_DETECTED, delta=HOAXES_DETECTED-290, delta_color="inverse")

    st.markdown("---")
    
    st.subheader("Real-Time User Activity & Hoaxes Detected ")
    
    # --- Simulated Time-Series Data for the Line Chart ---
    data_points = 8
    
    # Simulated active users growing trend
    base_users = np.linspace(500, 1200, data_points) + np.random.randint(-50, 50, data_points)
    
    # Simulated hoaxes detected
    base_hoaxes = np.linspace(300, 450, data_points) + np.random.randint(-30, 30, data_points)
    
    # Ensure all values are positive integers
    active_users = np.clip(base_users.astype(int), 0, None)
    hoaxes_detected = np.clip(base_hoaxes.astype(int), 0, None)
    
    chart_data = pd.DataFrame({
        "Active Users": active_users,
        "Hoaxes Detected": hoaxes_detected,
    }, index=pd.date_range("2024-01-01", periods=data_points, freq="W"))
    
    # Display the line chart
    st.line_chart(chart_data)

    st.markdown("---")
    
    # --- MODEL PERFORMANCE METRICS (From Original Code) ---
    st.header("2. Model Performance on Test Data")
    st.subheader("Classification Report")

    # Generate the classification report
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(columns=['support'], errors='ignore')

    st.dataframe(report_df, use_container_width=True)

    st.markdown("""
    - **Precision:** Out of all predicted 'Fake' items, how many were actually fake?
    - **Recall:** Out of all actual 'Fake' items, how many did the model correctly identify?
    - **F1-Score:** The harmonic mean of Precision and Recall.
    """)
    

def authenticated_app(tfidf_vectorizer, model, y_test, y_pred):
    """Handles routing for the authenticated user."""
    st.set_page_config(
        page_title="Fake News Detector",
        initial_sidebar_state="expanded",
        layout="wide" # Use wide layout for the analytics page
    )

    # Inject custom CSS for background image on the main app page
    st.markdown(
        f"""
        <style>
        /* --- START OF HEADER HIDING CSS --- */
        header {{
            visibility: hidden;
            display: none !important;
        }}
        
        /* This targets the main content area padding to prevent large gaps */
        .main {{
            padding-top: 0;
            padding-left: 0;
            padding-right: 0;
            padding-bottom: 0;
        }}
        /* This targets the container holding the menu and the deploy button */
        button[title="View source code"] {{ 
            display: none !important;
        }}
        .stDeployButton {{
            display: none !important;
        }}
        /* --- END OF HEADER HIDING CSS --- */
        
        .stApp {{
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Make main content area and sidebar slightly transparent for background to show */
        .main .block-container {{
            background-color: transparent; 
            
            border-radius: 0.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            max-width: 900px; /* Change width as you like */
            
            margin: auto;
        }}
        .css-vk32pt {{ /* This targets the sidebar background */
            background-color: rgba(240, 242, 246, 0.85); /* Default sidebar background with transparency */
        }}

        .stText, .stMarkdown, .stSubheader, .stTitle, .stLabel, h1, h2, h3 {{
            color: white; /* Darker text for contrast */
            text-shadow: 1px 1px 2px rgba(255,255,255,0.7); /* Subtle shadow for text on background */
        }}


        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Sidebar for User and Logout ---
    st.sidebar.header(f"👤 User: {st.session_state.username}")
    if st.sidebar.button("Log Out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
        
    st.sidebar.markdown("---")
    
    # --- Sidebar Navigation ---
    page = st.sidebar.radio("Select View", ("Detector", "Analytics"))

    st.sidebar.markdown("---")
    
    # --- Page Routing ---
    if page == "Detector":
        detector_page(tfidf_vectorizer, model)
    elif page == "Analytics":
        analytics_page(y_test, y_pred)


# --- 4. MAIN APPLICATION CONTROL FLOW ---

def main():
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    # Load resources outside the login check to prevent repeated training
    # We now get the test results as well
    tfidf_vectorizer, model, y_test, y_pred = train_model()

    if st.session_state.authenticated:
        # Show the main app if logged in, passing test results for analytics
        authenticated_app(tfidf_vectorizer, model, y_test, y_pred)
    else:
        # Show the login page otherwise
        login_page()

if __name__ == "__main__":
    main()
