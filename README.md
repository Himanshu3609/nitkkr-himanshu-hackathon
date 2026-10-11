# AI/NLP Risk Engine & Tactical Index Rebalancer - S&P Global & CRISIL Campus Hackathon
 
**Candidate Name:** Himanshu  
**College Email ID:** 124110002@nitkkr.ac.in  
**Personal Email ID:** himanshu36098@gmail.com
**College / Campus:** National Institute of Technology, Kurukshetra (NIT Kurukshetra)  
**Demo Video Link:** [YouTube / Unlisted Video Link]  
**Slide Deck Link :** [Google Slides / docs/presentation.pdf]  

---

## 1. Project Overview / Problem Statement & Approach

### Problem Statement
Financial institutions, asset managers, and risk analysts face an overwhelming deluge of unstructured data every second—spanning breaking market news, earnings releases, macroeconomic statements, and viral social media posts. Conventional risk management frameworks rely primarily on historical structured metrics (e.g., quarterly financial statements, trailing volatility), which creates a dangerous latency gap when unanticipated geopolitical or credit events strike. 

The objective of this case study is to architect and implement an **end-to-end AI/NLP Risk Engine** capable of parsing unstructured real-time feeds into structured, machine-readable risk intelligence (`Sentiment Score`, `Event Classification`, and `Impact Score (1-10)`).

### Approach & Downstream Implementation
To demonstrate a high-value real-world application of this intelligence, we implemented **Module A: Tactical Index Rebalancing**:
1. **Multi-Source Real-Time Ingestion**: Ingests unstructured inputs across four parallel vectors:
   - Live continuous RSS web crawling (**Google News Finance**, **MoneyControl Business**, **MoneyControl Markets**).
   - Real-time social media scraping (**X / Twitter** post URL and status ID parsing).
   - Historical multi-source datasets (**Kaggle financial news & tweets**).
   - Ad-hoc analyst manual text input.
2. **Unified NLP Risk Inference Engine**: Employs fine-tuned transformer architectures (**FinBERT** for domain-specific sentiment extraction and zero-shot **DistilBERT-MNLI** for multi-class event categorization across 14 financial categories), culminating in a non-linear **Impact Score (1-10)** modulated by event severity.
3. **Dynamic Tactical Portfolio Rebalancing**: Dynamically re-weights a mock stock portfolio (initial constituents: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`). When market events break, affected assets are identified (with **Dynamic Asset Discovery** that automatically integrates newly discovered tickers like `NVDA`, `WMT`, or `SPY`), rebalancing allocations tactically while maintaining a normalized 100% portfolio footprint.
4. **Advanced Factor & Forecast Analytics**: Features sentiment factor attribution (% contribution of events), company-wise net growth vs. fallout detection, and momentum-driven next-cycle AI allocation predictions.

---

## 2. Architecture & Tech Stack

### System Design & Data Flow
```
+-----------------------------------------------------------------------------------------+
|                                  1. DATA INGESTION LAYER                                |
+----------------------------+-----------------------------+------------------------------+
|   Live RSS Web Crawler     |    Live Social Media (X)    |    Kaggle Benchmark Feeds    |
|  - Google News (Finance)   |  - Real-time Tweet Scraper  |  - Financial News Sentiment  |
|  - MoneyControl (Business) |  - fxTwitter Embed Parsing  |  - Tweet Sentiment Dataset   |
|  - MoneyControl (Markets)  |  - Status ID & URL Parsing  |  - Automated Interleaving    |
+----------------------------+-----------------------------+------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------------+
|                                2. AI/NLP RISK ENGINE LAYER                              |
+-----------------------------------------------------------------------------------------+
|  [Text Normalization & Preprocessing] -> Raw Headline / Tweet / Social Post             |
|                                                                                         |
|  [A. Financial Sentiment Pipeline]                                                      |
|      * Model: ProsusAI/finbert (Domain-specific transformer fine-tuned on Financial     |
|               PhraseBank)                                                               |
|      * Output: Continuous Sentiment Score [-1.00 (Extremely Bearish) to +1.00 (Bullish)]|
|                                                                                         |
|  [B. Multi-Class Event Classification Pipeline]                                         |
|      * Model: typeform/distilbert-base-uncased-mnli (Zero-Shot NLI Classification)      |
|      * Granular Candidate Event Labels (14 Classes):                                    |
|        - Geopolitical Event          - Regulatory Action         - Credit Risk          |
|        - Macroeconomic Shift         - Supply Chain Disruption   - Product Launch       |
|        - Bankruptcy & Restructure    - Cybersecurity Breach      - Earnings Report      |
|        - Merger & Acquisition        - Leadership Change         - Legal / Litigation   |
|        - Dividend & Buyback          - ESG & Environmental Risk                         |
|                                                                                         |
|  [C. Non-Linear Impact Scoring Model]                                                   |
|      * Base: (|Sentiment Score| * 6.0) + (Classification Confidence * 4.0)               |
|      * High-Severity Multiplier: +2.5 boost for systemic risks (Geopolitical, Credit,   |
|        Macroeconomic, Bankruptcy, Regulatory, Supply Chain, Cybersecurity, Litigation)  |
|      * Scaled Output: Integer Risk Rating bounded from 1 (Minimal) to 10 (Critical)     |
+-----------------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------------+
|                     3. DOWNSTREAM MODULE A: TACTICAL INDEX REBALANCER                   |
+-----------------------------------------------------------------------------------------+
|  * Entity & Cashtag Dissector: Extracts $TICKER or maps corporate keywords              |
|  * Dynamic Asset Discovery: Automatically injects unindexed companies (e.g. WMT, NVDA)  |
|    into the portfolio universe with an initial baseline share (2.0%)                    |
|  * Rebalancing Formula: W_new = max(2.0%, W_curr + (Sentiment * (Impact / 10) * 10.0%)) |
|  * Sum-Normalization: Forces sum(W_all) = 100.0% preserving capital preservation bounds |
|  * Trajectory & Fallout Classifier: Analyzes weight delta slopes to flag                |
|    "Surging Momentum", "Steady Growth", "Selling Pressure", or "Severe Fallout"         |
|  * AI Next-Cycle Allocation Projection: Extrapolates momentum trend slopes to forecast  |
|    the next rebalancing target weights                                                  |
+-----------------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------------+
|                         4. INTERACTIVE STREAMLIT & PLOTLY DASHBOARD                     |
+-----------------------------------------------------------------------------------------+
|  [Section 1] Live Ingestion Controls (RSS Auto-Crawler, Tweet URL, Kaggle batch, manual)|
|  [Section 2] Live Feed Audit Table (Timestamp, Source, Headline, Ticker, Sent, Event)   |
|  [Section 3] Factor Attribution Analytics (% distribution donut & factor sentiment bar) |
|  [Section 4] Executive Performance Matrix (Initial vs Current weight, growth %, pattern)|
|  [Section 5] Interactive Tabbed Visualizations:                                         |
|              - Tab 1: Historical Spline Trajectory (with asset selector & highlight)    |
|              - Tab 2: 3-Way Allocation Shift (Initial vs Current vs AI Forecast)        |
|              - Tab 3: Current Index Donut Composition                                   |
+-----------------------------------------------------------------------------------------+
```

### Key Frameworks & Libraries
- **Language**: Python 3.10+ / 3.11
- **Deep Learning & NLP**:
  - `transformers` (Hugging Face pipelines for FinBERT & DistilBERT MNLI)
  - `torch` & `torchvision` (PyTorch acceleration backend)
- **Web Crawling & Data Ingestion**:
  - `feedparser` (Live RSS ingestion for Google News & MoneyControl)
  - `requests` & `beautifulsoup4` (X/Twitter embed API scraping & DOM parsing)
  - `kagglehub` (Automated historical benchmark downloading)
- **Data Engineering & Vectorization**:
  - `pandas` (Time-series matrix manipulation, state tracking, and tabular formatting)
  - `numpy` (Mathematical transformations and portfolio normalization)
- **Visualization & UI**:
  - `streamlit` (Reactive web application server with session state caching)
  - `plotly` (Interactive WebGL/SVG graphics, spline charts, and comparative distributions)

---

## 3. Dataset Used & Real-Time Feeds

### Sources
1. **Live Google News RSS Feed (Finance)**: Real-time RSS endpoint polling breaking global finance, earnings, and equity developments.
2. **Live MoneyControl RSS Feeds (Business & Market Reports)**: Real-time RSS endpoints pulling domestic and international market trends, commodities, and corporate filings.
3. **Real-Time X / Twitter Feed**: Public embed API integration extracting the literal post content from any public tweet URL or status ID.
4. **Historical Benchmark News Dataset**: `ankurzing/sentiment-analysis-for-financial-news` (Financial PhraseBank corpus containing financial news headlines).
5. **Historical Benchmark Tweet Dataset**: `thedevastator/tweet-sentiment-s-impact-on-stock-returns` (Historical financial social media sentiment).

### Assumptions
- Text feeds are primarily in English.
- Tweets and news headlines contain sufficient context for zero-shot NLI hypothesis verification.
- When an article refers to macroeconomic shifts without specifying a single corporation (e.g. Federal Reserve interest rate hikes), the system safely maps the signal to `SPY` (S&P 500 ETF proxy) rather than arbitrarily impacting an isolated company.
- Newly discovered companies enter with a protective 2.0% allocation floor, preventing extreme portfolio shocks.

---

## 4. Quickstart & Installation

The jury can run this application either **using Docker (Recommended for instant, zero-configuration evaluation)** or **natively using Python**.

---

### Option A: 1-Click Run via Docker (Recommended for Jury)

Prerequisites: Ensure **Docker** and **Docker Compose** are installed and running on your machine.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd nitkkr-himanshu-hackathon
   ```

2. **Run with Docker Compose (Single Command):**
   ```bash
   docker compose up --build
   ```
   *Or using standard Docker CLI:*
   ```bash
   docker build -t nitkkr-risk-engine .
   docker run -p 8501:8501 nitkkr-risk-engine
   ```

3. **Access the Application:**
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

---

### Option B: Native Python Installation

Prerequisites: Python 3.10, 3.11, or 3.12 installed on Windows, macOS, or Linux.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd nitkkr-himanshu-hackathon
   ```

2. **Create and activate a virtual environment:**
   - **On Windows (PowerShell / Command Prompt):**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **On macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Application:**
   ```bash
   streamlit run src/module_a_app.py
   ```
   *The dashboard will launch automatically in your default browser at `http://localhost:8501`.*

---

## 5. Key Results & Domain Impact

### What the Prototype Demonstrates
1. **Sub-Second Signal Transformation**: Converts messy, qualitative text into quantitative financial signals with zero human intervention.
2. **Quantitative Factor Attribution**: Dynamically aggregates the proportion of sentiment driven by distinct economic drivers (e.g., determining that 32% of bearish pressure is currently driven by Geopolitical Events vs. 14% by Credit Risk).
3. **Autonomous Dynamic Rebalancing**: Directly bridges NLP signals to asset weights—demonstrating real-time capital allocation adjustment based on breaking news flow.
4. **Asset Discovery & Growth Trajectory Matrix**: Automatically adapts to unknown assets appearing in the news, tracking their initial allocation, current weight, net growth, recent trajectory pattern (e.g., *Surging Momentum* vs. *Severe Fallout*), and AI-projected next-cycle target weight.
5. **Interactive Executive Visualization**: Offers a clean, tabbed analytical dashboard equipped with audit logs, multi-asset comparative bar charts, and spline curve trend analysis with company highlighting.

### Business & Quantitative Finance Impact
In high-frequency and quantitative asset management, milliseconds define profitability. Traditional hedge funds often suffer from information latency when processing text. By converting streaming social media and news feeds into structured risk vectors, quantitative funds can:
- **Hedge Tail Risk Automatically**: Instantly de-risk allocations before credit defaults or regulatory sanctions reflect in trailing balance sheets.
- **Capture Sentiment Alpha**: Scale into companies experiencing positive product momentum or earnings beats the exact second headlines publish.
- **Maintain Rigorous Capital Preservation**: Enforce strict minimum/maximum allocation guardrails, preventing automated over-concentration in volatile assets.

---

## 6. Repository Structure
```
nitkkr-himanshu-hackathon/
├── README.md                # Comprehensive documentation, setup, architecture & results
├── Dockerfile               # Production container definition for jury evaluation
├── docker-compose.yml       # 1-click Docker Compose configuration
├── .dockerignore            # Optimizes Docker build context & image layer caching
├── requirements.txt         # Production Python dependencies
├── LICENSE                  # Open-source MIT License
├── src/
│   ├── module_a_app.py          # Primary Streamlit application & interactive dashboard
│   ├── risk_engine_pipeline.py  # Core AI/NLP Risk Engine (FinBERT + DistilBERT MNLI)
│   └── risk_data_ingestion.py   # Multi-source ingestion & Kaggle data streamer
├── data/                    # Local storage for sample datasets & offline benchmarks
└── docs/
    ├── presentation.pdf     # Submission slide deck
    └── architecture.png     # High-resolution architectural diagram
```
