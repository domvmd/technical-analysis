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
@st.cache_data(ttl=3600, show_spinner="Loading stock data...")
def fetch_stock_data(ticker, period="1y"):
    """Fetch historical stock data with dynamic interval selection"""
    try:
        interval_rules = {
            "1d": "60m",
            "5d": "60m",
            "1mo": "60m",
            "3mo": "1d",
            "6mo": "1d",
            "1y": "1d",
            "5y": "1d"
        }
        
        valid_periods = list(interval_rules.keys())
        period = period if period in valid_periods else "1d"
        interval = interval_rules[period]
        
        # Critical fix: Remove group_by parameter
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            timeout=10
        )
        
        if df.empty:
            st.error(f"No data found for {ticker}. Check ticker symbol.")
            return None
            
        # Ensure standard column names
        df.columns = df.columns.str.lower()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        # Filter market hours for intraday data
        df.index = pd.to_datetime(df.index)
        if interval.endswith('m'):
            df = df.between_time('09:30', '16:00')
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Cache the calculate_technical_indicators function
@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    try:
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Calculate indicators
        close = df['close']
        
        # Moving averages
        df["ma20"] = close.rolling(window=20).mean() if len(df) >= 20 else np.nan
        df["ma50"] = close.rolling(window=50).mean() if len(df) >= 50 else np.nan

        # RSI calculation
        delta = close.diff().dropna()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df.dropna()
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return pd.DataFrame()

# Function to analyze candlestick patterns using OpenAI
def analyze_candlestick_patterns(client, stock_data, period):
    """Analyze candlestick patterns using OpenAI"""
    try:
        data = stock_data.tail(60)  # Analyze last 60 periods

        description = (
            f"The stock data for the selected period ({period}) shows the following candlestick patterns:\n"
            f"- Open: {data['Open'].values[-5:]}\n"  # Show last 5 values
            f"- High: {data['High'].values[-5:]}\n"
            f"- Low: {data['Low'].values[-5:]}\n"
            f"- Close: {data['Close'].values[-5:]}\n"
            f"- Volume: {data['Volume'].values[-5:]}\n"
            f"Please analyze ONLY TOP 5 significant **candlestick patterns** and provide insights considering: "
            f"\n1. Pattern strength and confirmation"
            f"\n2. Confluence with RSI/MA/Volume"
            f"\n3. Recent price action context"
            f"Each candlestick represents a specific time interval (e.g., 1 hour or 1 day). "
            f"Refer to them as '1st candlestick', '2nd candlestick', etc., instead of '1st hour' or '17th hour'. "
            f"At the end of the summary, do a price prediction after 4, 8, and 12 months. "
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Chartered Market Technician with more than 20 years experience. "
                    "Analyze the **candlestick patterns** and provide insights only on the significant ones."
                    "Identify MAXIMUM 5 most significant patterns."
                    "For each pattern: Name and location (e.g., '3rd candlestick: Bullish Engulfing'), Confidence level (High/Medium/Low), Key confirmation factors (volume, indicator alignment), and Immediate price implications)"
                    "Trading plan: Clear entry/exit levels: Ideal Buy Zone: ${X} - ${Y}, Stop Loss: ${Z}, Take profit: ${A} (short term); Risk-reward ratio"
                    "Give a suggestion on what **prices** to buy or sell the stock."
                    "Refer to each candlestick as '1st candlestick', '2nd candlestick', etc., instead of '1st hour' or '17th hour'."
                    "At the end of the summary, do a price prediction after 4, 8, and 12 months."
                    "Professional Tone: Avoid speculation, highlight key support/resistance, and mention any divergence patterns.",
                },
                {"role": "user", "content": description},
            ],
            temperature=0.3,            
        )

        analysis = response.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        st.error(f"Error in candlestick pattern analysis: {str(e)}")
        return "Unable to analyze candlestick patterns."

# Function to plot predictions using candlestick chart
def plot_predictions(stock_data, prediction, period):
    """Create visualization of stock data and predictions using candlestick chart
    Returns:
        tuple: (candlestick_img_path, rsi_img_path)
    """
    try:
        data = stock_data

        # Create a candlestick chart
        mpf.plot(
            data,
            type="candle",
            style="yahoo",  # You can choose other styles like 'classic', 'yahoo', etc.
            title=f"{prediction['ticker']} Stock Analysis - {prediction['prediction_date']}",
            ylabel="Price ($)",  # Updated to $ for US stocks
            volume=True,  # Add volume subplot
            mav=(20, 50),  # Add 20-hour and 50-hour moving averages
            figsize=(14, 8),
            show_nontrading=False,
            returnfig=False,
        )

        # Display the plot in Streamlit
        st.pyplot(plt.gcf())

        # Save the candlestick chart
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches="tight", dpi=300)
            candlestick_img = tmpfile.name

        # Create RSI plot with improved styling
        plt.figure(figsize=(14, 4))
        plt.plot(data.index, data["RSI"], label="RSI", color="purple", linewidth=2)

        # Add RSI zones with better visibility
        plt.axhline(
            y=70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)"
        )
        plt.axhline(
            y=30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)"
        )

        plt.fill_between(
            data.index,
            data["RSI"],
            70,
            where=(data["RSI"] >= 70),
            color="red",
            alpha=0.2,
        )
        plt.fill_between(
            data.index,
            data["RSI"],
            30,
            where=(data["RSI"] <= 30),
            color="green",
            alpha=0.2,
        )

        plt.title("RSI Indicator", fontsize=12, pad=20)
        plt.ylabel("RSI", fontsize=10)
        plt.xlabel("Date", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis="x", rotation=45)

        # Move RSI legend to the right
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=10,
        )

        # Display the RSI plot in Streamlit
        st.pyplot(plt)

        # Save the RSI chart
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches="tight", dpi=300)
            rsi_img = tmpfile.name

    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")
        return None, None

    return candlestick_img, rsi_img


def generate_pdf_report(prediction, analysis):
    """Generate a PDF report with the analysis results"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Function to sanitize text for PDF
    def sanitize_text(text):
        # Replace unsupported Unicode characters with ASCII equivalents
        replacements = {
            "–": "-",  # Replace en dash with hyphen
            "—": "-",  # Replace em dash with hyphen
            "“": '"',  # Replace left double quotation mark with ASCII quote
            "”": '"',  # Replace right double quotation mark with ASCII quote
            "‘": "'",  # Replace left single quotation mark with ASCII quote
            "’": "'",  # Replace right single quotation mark with ASCII quote
            "≈": "≈",  # Approximately equal
            "≠": "!=",  # Not equal
            "≤": "<=",  # Less than or equal
            "≥": ">=",  # Greater than or equal
            "±": "+/-",  # Plus-minus
            "°": "deg",  # Degree symbol
            "•": "*",  # Bullet point
            "…": "...",  # Ellipsis
            "→": "->",  # Right arrow
            "←": "<-",  # Left arrow
            "×": "x",  # Multiplication sign
            "÷": "/",  # Division sign
            "∞": "inf",  # Infinity
            "µ": "u",  # Micro symbol
            "€": "EUR",  # Euro symbol
            "£": "GBP",  # Pound symbol
            "¥": "JPY",  # Yen symbol
            "©": "(c)",  # Copyright
            "®": "(R)",  # Registered trademark
            "™": "(TM)",  # Trademark
            "§": "Sect.",  # Section symbol
            "¶": "P.",  # Paragraph symbol
        }
        # First try to encode as latin-1
        try:
            return text.encode("latin-1", "replace").decode("latin-1")
        except:
            # If that fails, replace remaining special characters
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text.encode("ascii", "replace").decode("ascii")

    # Add title
    pdf.cell(200, 10, txt=sanitize_text("Stock Analysis Report"), ln=True, align="C")

    # Add prediction details
    pdf.cell(200, 10, txt=sanitize_text(f"Ticker: {prediction['ticker']}"), ln=True)
    pdf.cell(
        200,
        10,
        txt=sanitize_text(f"Current Price: ${prediction['last_close']:.2f}"),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(f"Predicted Price: ${prediction['predicted_price']:.2f}"),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(
            f"Predicted Change: {((prediction['predicted_price'] / prediction['last_close']) - 1) * 100:.1f}%"
        ),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(f"Prediction Date: {prediction['prediction_date']}"),
        ln=True,
    )

    # Add technical indicators
    pdf.cell(200, 10, txt=sanitize_text("Technical Indicators:"), ln=True)
    pdf.cell(
        200,
        10,
        txt=sanitize_text(
            f"20-hour MA: ${prediction['technical_indicators']['ma20']:.2f}"
        ),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(
            f"50-hour MA: ${prediction['technical_indicators']['ma50']:.2f}"
        ),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(f"RSI: {prediction['technical_indicators']['rsi']:.2f}"),
        ln=True,
    )

    # Add market insight
    pdf.cell(200, 10, txt=sanitize_text("Market Insight:"), ln=True)
    pdf.cell(
        200,
        10,
        txt=sanitize_text(f"Summary: {prediction['market_insight']['summary']}"),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(f"Risk Level: {prediction['market_insight']['risk_level']}"),
        ln=True,
    )
    pdf.cell(
        200,
        10,
        txt=sanitize_text(
            f"Recommendation: {prediction['market_insight']['recommendation']}"
        ),
        ln=True,
    )

    # Add candlestick pattern analysis
    pdf.cell(200, 10, txt=sanitize_text("Candlestick Pattern Analysis:"), ln=True)
    pdf.multi_cell(0, 10, txt=sanitize_text(analysis))

    # Save the PDF
    pdf_output = f"{prediction['ticker']}_analysis_report.pdf"
    pdf.output(pdf_output)
    return pdf_output


# Function to predict the next day's stock price
def predict_next_day(ticker, period):
    """Generate stock prediction"""
    try:
        stock_data = fetch_stock_data(ticker, period=period)
        if stock_data is None or stock_data.empty:
            return None, None

        stock_data = calculate_technical_indicators(stock_data)
        if stock_data.empty:
            return None, None

        # Validate required columns for prediction
        if 'close' not in stock_data.columns:
            st.error("Missing 'close' price data")
            return None, None

        # Rest of prediction logic...
        close = stock_data['close']
        returns = close.pct_change().tail(5)
        avg_return = returns.mean()
        last_close = float(close.iloc[-1])
        predicted_price = last_close * (1 + avg_return)
        
        # Calculate sentiment analysis
        sentiment_analysis = calculate_sentiment_analysis(stock_data)

        # Prepare result
        result = {
            "ticker": ticker,
            "predicted_price": predicted_price,
            "last_close": last_close,
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            "sentiment_analysis": sentiment_analysis["sentiment_score"],
            "market_insight": {
                "summary": f"Analysis based on technical indicators and recent price movements.",
                "risk_level": sentiment_analysis["risk_level"],
                "recommendation": sentiment_analysis["recommendation"],
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


# Perform Sentiment Analysis
def calculate_sentiment_analysis(stock_data):
    """Calculate sentiment score, risk level, and recommendation"""
    try:
        # Get closing prices and volume
        closing_prices = stock_data["Close"].values
        volumes = stock_data["Volume"].values

        # Calculate price trend
        initial_price = closing_prices[0]
        current_price = closing_prices[-1]
        price_change = (current_price - initial_price) / initial_price
        price_sentiment = max(-1, min(1, price_change))  # Clamp to [-1, 1]

        # Calculate volume trend
        initial_volume = volumes[0]
        current_volume = volumes[-1]
        volume_change = (current_volume - initial_volume) / initial_volume
        volume_sentiment = max(-1, min(1, volume_change))  # Clamp to [-1, 1]

        # Calculate RSI sentiment
        rsi = stock_data["RSI"].iloc[-1]
        rsi_sentiment = 0
        if rsi > 70:
            rsi_sentiment = -1  # Bearish
        elif rsi < 30:
            rsi_sentiment = 1  # Bullish

        # Calculate moving average sentiment
        ma20 = stock_data["MA20"].iloc[-1]
        ma50 = stock_data["MA50"].iloc[-1]
        ma_sentiment = 1 if ma20 > ma50 else -1

        # Calculate sentiment score
        sentiment_score = (
            0.4 * price_sentiment
            + 0.2 * volume_sentiment
            + 0.2 * rsi_sentiment
            + 0.2 * ma_sentiment
        )

        # Calculate risk level
        volatility = np.std(closing_prices)
        max_volatility = np.max(closing_prices) - np.min(closing_prices)
        normalized_volatility = min(1, volatility / max_volatility)  # Normalize

        rsi_risk = 0
        if rsi > 70:
            rsi_risk = 1  # High risk
        elif rsi < 30:
            rsi_risk = 0  # Low risk
        else:
            rsi_risk = 0.5  # Medium risk

        risk_level = 0.6 * normalized_volatility + 0.4 * rsi_risk

        # Classify risk level
        if risk_level < 0.3:
            risk_level_str = "LOW"
        elif 0.3 <= risk_level < 0.7:
            risk_level_str = "MEDIUM"
        else:
            risk_level_str = "HIGH"

        # Generate recommendation
        if sentiment_score > 0.5 and risk_level_str == "LOW":
            recommendation = "STRONG BUY"
        elif sentiment_score > 0.5 and risk_level_str == "MEDIUM":
            recommendation = "BUY"
        elif sentiment_score > 0.5 and risk_level_str == "HIGH":
            recommendation = "HOLD"
        elif sentiment_score < -0.5 and risk_level_str == "LOW":
            recommendation = "SELL"
        elif sentiment_score < -0.5 and risk_level_str == "MEDIUM":
            recommendation = "HOLD"
        elif sentiment_score < -0.5 and risk_level_str == "HIGH":
            recommendation = "STRONG SELL"
        else:
            recommendation = "HOLD"

        # Prepare result
        result = {
            "sentiment_score": sentiment_score,
            "risk_level": risk_level_str,
            "recommendation": recommendation,
        }
        return result

    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return {
            "sentiment_score": 0,
            "risk_level": "MEDIUM",
            "recommendation": "HOLD",
        }


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
                    st.write(
                        f"**Predicted Price:** ${prediction['predicted_price']:.2f}"
                    )
                    st.write(
                        f"**Predicted Change:** {((prediction['predicted_price'] / prediction['last_close']) - 1) * 100:.1f}%"
                    )
                    st.write(f"**Prediction Date:** {prediction['prediction_date']}")

                    st.subheader("Technical Indicators")
                    st.write(
                        f"**20-hour MA:** ${prediction['technical_indicators']['ma20']:.2f}"
                    )
                    st.write(
                        f"**50-hour MA:** ${prediction['technical_indicators']['ma50']:.2f}"
                    )
                    st.write(
                        f"**RSI:** {prediction['technical_indicators']['rsi']:.2f}"
                    )

                    st.subheader("Market Insight")
                    st.write(f"**Summary:** {prediction['market_insight']['summary']}")
                    st.write(
                        f"**Risk Level:** {prediction['market_insight']['risk_level']}"
                    )
                    st.write(
                        f"**Recommendation:** {prediction['market_insight']['recommendation']}"
                    )

                    # Clear session state for report files when analyzing a new stock or period
                    if "report_files" in st.session_state:
                        del st.session_state.report_files

                    # Analyze candlestick patterns using AI first
                    analysis = analyze_candlestick_patterns(client, stock_data, period)
                    st.subheader("Candlestick Pattern Analysis")
                    st.write(analysis)

                    # Plot predictions and get image paths
                    candlestick_img, rsi_img = plot_predictions(
                        stock_data, prediction, period
                    )

                    # Generate PDF report
                    pdf_report = generate_pdf_report(prediction, analysis)

                    # Store file paths in session state
                    st.session_state.report_files = {
                        "pdf_report": pdf_report,
                        "candlestick_img": candlestick_img,
                        "rsi_img": rsi_img,
                    }

                    # Create ZIP file with all reports
                    zip_filename = f"{ticker}_analysis_reports.zip"
                    with zipfile.ZipFile(zip_filename, "w") as zipf:
                        # Add PDF report
                        if st.session_state.report_files["pdf_report"]:
                            zipf.write(
                                st.session_state.report_files["pdf_report"],
                                os.path.basename(
                                    st.session_state.report_files["pdf_report"]
                                ),
                            )

                        # Add candlestick chart
                        if st.session_state.report_files["candlestick_img"]:
                            zipf.write(
                                st.session_state.report_files["candlestick_img"],
                                f"{ticker}_candlestick_chart.png",
                            )

                        # Add RSI chart
                        if st.session_state.report_files["rsi_img"]:
                            zipf.write(
                                st.session_state.report_files["rsi_img"],
                                f"{ticker}_rsi_chart.png",
                            )

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
                    st.error(
                        f"Unable to analyze {ticker}. Please check the ticker symbol."
                    )


# Run the Streamlit app
if __name__ == "__main__":
    main()
