# Import necessary libraries
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import mplfinance as mpf
import warnings
import json
import streamlit as st
import time
from fpdf import FPDF
import tempfile
import zipfile

warnings.filterwarnings("ignore")

# Set seaborn style
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

# Configure plot settings
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# Load environment variables
load_dotenv()

# Initialize OpenAI client
def initialize_openai_client(api_key):
    """Initialize the OpenAI client with the provided API key."""
    if not api_key:
        st.error("Please enter a valid Deepseek API key.")
        return None
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

# Cache the fetch_stock_data function to avoid redundant API calls
@st.cache_data
def fetch_stock_data(ticker, period="1y", interval="1d"):
    """Fetch historical stock data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        # Check if the fetched data has at least 50 data points
        if len(df) < 50:
            df = stock.history(period="3mo", interval="1d")

        if df.empty:
            st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol.")
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Cache the calculate_technical_indicators function
@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators with forward fill"""
    try:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill()
        
        # Moving averages
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        
        # RSI calculation
        delta = df["Close"].diff().ffill()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan).ffill()
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

# Function to analyze candlestick patterns using OpenAI
def analyze_candlestick_patterns(client, stock_data, period):
    """Analyze candlestick patterns using OpenAI with expert technical analysis"""
    try:
        # Get data for the selected period
        period_map = {
            "1d": 24, "5d": 120, "1mo": 30,
            "6mo": 180, "1y": 365, "5y": 1825
        }
        data = stock_data.tail(period_map.get(period, 30))

        # Enhanced description with technical indicators
        description = (
            f"Stock data for {period} period:\n"
            f"Latest 10 data points:\n"
            f"{data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()}\n\n"
            f"Technical Indicators:\n"
            f"- 20-period MA: {data['MA20'].iloc[-1]:.2f}\n"
            f"- 50-period MA: {data['MA50'].iloc[-1]:.2f}\n"
            f"- RSI: {data['RSI'].iloc[-1]:.2f}\n\n"
            f"Analyze TOP 5 significant candlestick patterns considering:\n"
            f"1. Pattern strength and confirmation\n"
            f"2. Confluence with RSI/MA/Volume\n"
            f"3. Recent price action context"
        )

        # Expert system prompt
        expert_prompt = """You're a Chartered Market Technician (CMT) with 20 years experience. Analyze strictly following these rules:

1. Pattern Analysis:
- Identify MAXIMUM 5 most significant patterns
- For each pattern:
  * Name & location (e.g., '3rd candlestick: Bullish Engulfing')
  * Confidence level (High/Medium/Low)
  * Key confirmation factors (volume, indicator alignment)
  * Immediate price implications

2. Trading Plan:
- Clear entry/exit levels:
  * Ideal Buy Zone: ${X} - ${Y}
  * Stop Loss: ${Z}
  * Take Profit: ${A} (short-term)
- Risk-reward ratio

3. Price Predictions (technical-only):
- 4 months: 
- 8 months: 
- 12 months: 
Format predictions with price ranges and confidence percentages

4. Professional Tone:
- Avoid speculation
- Highlight key support/resistance
- Mention any divergence patterns"""

        # Send to OpenAI
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": expert_prompt},
                {"role": "user", "content": description}
            ],
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return "Analysis unavailable"

# Function to plot predictions using candlestick chart
def plot_predictions(stock_data, prediction, period):
    """Create visualization of stock data and predictions using candlestick chart"""
    try:
        period_map = {
            "1d": 24, "5d": 120, "1mo": 30,
            "6mo": 180, "1y": 365, "5y": 1825
        }
        data = stock_data.tail(period_map.get(period, 30))

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3,1]},
                                     sharex=True)
        
        # Plot candlestick chart
        mpf.plot(data, type='candle', style='yahoo',
                 mav=(20,50), volume=ax2, ax=ax1,
                 show_nontrading=False,
                 axtitle=f"{prediction['ticker']} Technical Analysis")

        # Add RSI plot
        ax1_rsi = ax1.twinx()
        ax1_rsi.plot(data['RSI'], color='purple', alpha=0.5, linewidth=1.5)
        ax1_rsi.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax1_rsi.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax1_rsi.set_ylabel('RSI', color='purple')

        # Customize plot appearance
        ax1.set_ylabel("Price ($)")
        ax2.set_ylabel("Volume")
        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)

        # Save plot
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches="tight", dpi=300)
            candlestick_img = tmpfile.name

        return candlestick_img, None

    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")
        return None, None

# Rest of the functions (generate_pdf_report, predict_next_day, 
# calculate_sentiment_analysis, and main) remain the same as in your original code
# ... [Keep the rest of your code unchanged from the original version] ...

# Streamlit App
def main():
    st.title("Papa Dom's Stock/Crypto Price Analysis App")
    st.write("Analyze stocks from the US and other markets or Cryptocurrency.")

    # Input Deepseek API key
    deepseek_api_key = st.text_input(
        "Enter your Deepseek API key:",
        type="password",
        help="You can find your API key at https://platform.deepseek.com/account/api-keys",
    )

    # Initialize OpenAI client
    client = initialize_openai_client(deepseek_api_key)

    # Input ticker symbol
    ticker = st.text_input(
        "Enter a ticker compatible with Yahoo Finance (e.g., NVDA for Nvidia, BTC-USD for Bitcoin):",
        "AAPL",
    )

    # Time period selection
    period = st.selectbox(
        "Select the time period for analysis:",
        options=["1d", "5d", "1mo", "6mo", "1y", "5y"],
        index=4,  # Default to 1 year
    )

    # Button to analyze the stock
    if st.button("Analyze"):
        if not client:
            st.error("Please enter a valid Deepseek API key to proceed.")
        else:
            with st.spinner("Analyzing stock data..."):
                time.sleep(1)
                prediction, stock_data = predict_next_day(ticker, period)

                if prediction and stock_data is not None:
                    # Display results
                    st.subheader("Stock Analysis Results")
                    cols = st.columns(2)
                    cols[0].write(f"**Ticker:** {prediction['ticker']}")
                    cols[1].write(f"**Current Price:** ${prediction['last_close']:.2f}")
                    cols[0].write(f"**Predicted Price:** ${prediction['predicted_price']:.2f}")
                    cols[1].write(f"**Predicted Change:** {((prediction['predicted_price'] / prediction['last_close']) - 1) * 100:.1f}%")
                    st.write(f"**Prediction Date:** {prediction['prediction_date']}")

                    st.subheader("Technical Indicators")
                    tech_cols = st.columns(3)
                    tech_cols[0].write(f"**20-period MA:** ${prediction['technical_indicators']['ma20']:.2f}")
                    tech_cols[1].write(f"**50-period MA:** ${prediction['technical_indicators']['ma50']:.2f}")
                    tech_cols[2].write(f"**RSI:** {prediction['technical_indicators']['rsi']:.2f}")

                    st.subheader("Market Insight")
                    insight_cols = st.columns(3)
                    insight_cols[0].write(f"**Summary:** {prediction['market_insight']['summary']}")
                    insight_cols[1].write(f"**Risk Level:** {prediction['market_insight']['risk_level']}")
                    insight_cols[2].write(f"**Recommendation:** {prediction['market_insight']['recommendation']}")

                    # Clear session state for report files
                    if "report_files" in st.session_state:
                        del st.session_state.report_files

                    # Analyze candlestick patterns
                    analysis = analyze_candlestick_patterns(client, stock_data, period)
                    st.subheader("Candlestick Pattern Analysis")
                    st.write(analysis)

                    # Plot predictions
                    candlestick_img, rsi_img = plot_predictions(stock_data, prediction, period)

                    # Generate PDF report
                    pdf_report = generate_pdf_report(prediction, analysis)

                    # Create ZIP file with all reports
                    zip_filename = f"{ticker}_analysis_reports.zip"
                    with zipfile.ZipFile(zip_filename, "w") as zipf:
                        zipf.write(pdf_report, os.path.basename(pdf_report))
                        if candlestick_img:
                            zipf.write(candlestick_img, f"{ticker}_candlestick.png")

                    # Download button
                    st.subheader("Download All Reports")
                    with open(zip_filename, "rb") as f:
                        st.download_button(
                            label="📦 Download All Reports (ZIP)",
                            data=f,
                            file_name=zip_filename,
                            mime="application/zip",
                        )

                    if st.button("Analyze a Different Stock"):
                        st.session_state.clear()
                        st.rerun()
                else:
                    st.error(f"Unable to analyze {ticker}. Please check the ticker symbol.")

if __name__ == "__main__":
    main()
