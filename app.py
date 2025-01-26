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
from fpdf import FPDF  # For generating PDF reports
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
            # If not, fetch more data by extending the period
            extended_period = "3mo"  # Default to 3 months if insufficient data
            df = stock.history(period=extended_period, interval="1d")

        if df.empty:
            st.error(
                f"No data found for ticker: {ticker}. Please check the ticker symbol."
            )
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
        
        return df.dropna()  # Remove initial NaN rows
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df


# Function to analyze candlestick patterns using OpenAI
def analyze_candlestick_patterns(client, stock_data, period):
    """Analyze candlestick patterns using OpenAI with expert technical analysis"""
    try:
        # Get data for the selected period
        if period == "1d":
            data = stock_data.tail(24)  # 24 hours for 1 day
        elif period == "5d":
            data = stock_data.tail(120)  # 24 hours * 5 days = 120 hours
        elif period == "1mo":
            data = stock_data.tail(30)
        elif period == "6mo":
            data = stock_data.tail(180)
        elif period == "1y":
            data = stock_data.tail(365)
        elif period == "5y":
            data = stock_data.tail(1825)
        else:
            data = stock_data.tail(30)  # Default to 30 days

        # Enhanced description with technical indicators
        description = (
            f"Stock data for {period} period:\n"
            f"- Open/High/Low/Close/Volume: Latest 10 values shown\n"
            f"{data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()}\n\n"
            f"Technical Indicators:\n"
            f"- 20-period MA: {data['MA20'].iloc[-1]:.2f}\n"
            f"- 50-period MA: {data['MA50'].iloc[-1]:.2f}\n"
            f"- RSI: {data['RSI'].iloc[-1]:.2f}\n\n"
            f"Analyze ONLY TOP 5 significant candlestick patterns considering:"
            f"\n1. Pattern strength and confirmation"
            f"\n2. Confluence with RSI/MA/Volume"
            f"\n3. Recent price action context"
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
        # Get data for the selected period
        if period == "1d":
            data = stock_data.tail(24)
        elif period == "5d":
            data = stock_data.tail(120)
        elif period == "1mo":
            data = stock_data.tail(30)
        elif period == "6mo":
            data = stock_data.tail(180)
        elif period == "1y":
            data = stock_data.tail(365)
        elif period == "5y":
            data = stock_data.tail(1825)
        else:
            data = stock_data.tail(30)

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
        plt.close(fig)  # Prevent figure accumulation

        # Save plots and return paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches="tight", dpi=300)
            candlestick_img = tmpfile.name

        return candlestick_img, None  # Return RSI image path if needed

    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")
        return None, None


# Function to generate PDF report
def generate_pdf_report(prediction, analysis):
    """Generate a PDF report with the analysis results"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.cell(200, 10, txt="Stock Analysis Report", ln=True, align="C")

    # Add prediction details
    pdf.cell(200, 10, txt=f"Ticker: {prediction['ticker']}", ln=True)
    pdf.cell(200, 10, txt=f"Current Price: ${prediction['last_close']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Price: ${prediction['predicted_price']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Change: {((prediction['predicted_price'] / prediction['last_close']) - 1) * 100:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Date: {prediction['prediction_date']}", ln=True)

    # Add technical indicators
    pdf.cell(200, 10, txt="Technical Indicators:", ln=True)
    pdf.cell(200, 10, txt=f"20-hour MA: ${prediction['technical_indicators']['ma20']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"50-hour MA: ${prediction['technical_indicators']['ma50']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"RSI: {prediction['technical_indicators']['rsi']:.2f}", ln=True)

    # Add market insight
    pdf.cell(200, 10, txt="Market Insight:", ln=True)
    pdf.cell(200, 10, txt=f"Summary: {prediction['market_insight']['summary']}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {prediction['market_insight']['risk_level']}", ln=True)
    pdf.cell(200, 10, txt=f"Recommendation: {prediction['market_insight']['recommendation']}", ln=True)

    # Add candlestick pattern analysis
    pdf.cell(200, 10, txt="Candlestick Pattern Analysis:", ln=True)
    pdf.multi_cell(0, 10, txt=analysis)

    # Save the PDF
    pdf_output = f"{prediction['ticker']}_analysis_report.pdf"
    pdf.output(pdf_output)
    return pdf_output


# Function to predict the next day's stock price
def predict_next_day(ticker, period):
    """Generate stock prediction"""
    try:
        # Fetch and store stock data
        interval = "1h" if period in ["1d", "5d"] else "1d"
        stock_data = fetch_stock_data(ticker, period=period, interval=interval)
        if stock_data is None:
            return None, None

        # Calculate technical indicators
        stock_data = calculate_technical_indicators(stock_data)

        # Calculate prediction using simple model
        returns = stock_data["Close"].pct_change().tail(5)
        avg_return = returns.mean()
        last_close = float(stock_data["Close"].iloc[-1])
        predicted_price = last_close * (1 + avg_return)

        # Prepare result
        result = {
            "ticker": ticker,
            "predicted_price": predicted_price,
            "last_close": last_close,
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "market_insight": {
                "summary": "Analysis based on technical indicators and recent price movements.",
                "risk_level": "LOW",
                "recommendation": "HOLD",
            },
            "technical_indicators": {
                "ma20": float(stock_data["MA20"].iloc[-1]),
                "ma50": float(stock_data["MA50"].iloc[-1]),
                "rsi": float(stock_data["RSI"].iloc[-1]),
            },
        }

        return result, stock_data

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None


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
                time.sleep(1)  # Simulate a delay for demonstration
                prediction, stock_data = predict_next_day(ticker, period)

                if prediction and stock_data is not None:
                    # Display results
                    st.subheader("Stock Analysis Results")
                    st.write(f"**Ticker:** {prediction['ticker']}")
                    st.write(f"**Current Price:** ${prediction['last_close']:.2f}")
                    st.write(f"**Predicted Price:** ${prediction['predicted_price']:.2f}")
                    st.write(f"**Predicted Change:** {((prediction['predicted_price'] / prediction['last_close']) - 1) * 100:.1f}%")
                    st.write(f"**Prediction Date:** {prediction['prediction_date']}")

                    st.subheader("Technical Indicators")
                    st.write(f"**20-hour MA:** ${prediction['technical_indicators']['ma20']:.2f}")
                    st.write(f"**50-hour MA:** ${prediction['technical_indicators']['ma50']:.2f}")
                    st.write(f"**RSI:** {prediction['technical_indicators']['rsi']:.2f}")

                    st.subheader("Market Insight")
                    st.write(f"**Summary:** {prediction['market_insight']['summary']}")
                    st.write(f"**Risk Level:** {prediction['market_insight']['risk_level']}")
                    st.write(f"**Recommendation:** {prediction['market_insight']['recommendation']}")

                    # Analyze candlestick patterns using AI first
                    analysis = analyze_candlestick_patterns(client, stock_data, period)
                    st.subheader("Candlestick Pattern Analysis")
                    st.write(analysis)

                    # Plot predictions and get image paths
                    candlestick_img, rsi_img = plot_predictions(stock_data, prediction, period)

                    # Generate PDF report
                    pdf_report = generate_pdf_report(prediction, analysis)

                    # Create ZIP file with all reports
                    zip_filename = f"{ticker}_analysis_reports.zip"
                    with zipfile.ZipFile(zip_filename, "w") as zipf:
                        zipf.write(pdf_report, os.path.basename(pdf_report))
                        if candlestick_img:
                            zipf.write(candlestick_img, f"{ticker}_candlestick_chart.png")

                    # Single download button for ZIP file
                    st.subheader("Download All Reports")
                    with open(zip_filename, "rb") as f:
                        st.download_button(
                            label="📦 Download All Reports (ZIP)",
                            data=f,
                            file_name=zip_filename,
                            mime="application/zip",
                        )

                    if st.button("Analyze a Different Stock"):
                        # Clear session state to reset the app
                        st.session_state.clear()
                        st.rerun()  # Rerun the app to reset the UI
                else:
                    st.error(f"Unable to analyze {ticker}. Please check the ticker symbol.")


# Run the Streamlit app
if __name__ == "__main__":
    main()
