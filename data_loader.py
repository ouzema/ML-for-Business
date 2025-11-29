import pandas as pd
import numpy as np
import yfinance as yf
import time
import re
from collections import Counter

def load_and_filter_data(filepath):
    """Loads data, filters for Eurozone, and samples."""
    print("Loading data...")
    df = pd.read_excel(filepath, header=1)
    
    # Eurozone countries (as of 2024)
    eurozone_countries = [
        "Austria", "Belgium", "Croatia", "Cyprus", "Estonia", "Finland", "France", 
        "Germany", "Greece", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", 
        "Malta", "Netherlands", "Portugal", "Slovakia", "Slovenia", "Spain"
    ]
    
    # Filter for Eurozone targets
    # Column name: 'Country/Region of Incorporation [Target/Issuer]'
    # Check for exact matches or partial matches if needed. 
    # Data inspection showed standard country names, but let's be safe.
    
    target_country_col = 'Country/Region of Incorporation [Target/Issuer]'
    
    # Normalize country names just in case (strip whitespace)
    df[target_country_col] = df[target_country_col].astype(str).str.strip()
    
    print(f"Total transactions before filtering: {len(df)}")
    
    # Filter
    df_euro = df[df[target_country_col].isin(eurozone_countries)].copy()
    print(f"Transactions in Eurozone: {len(df_euro)}")
    
    # Use ALL Eurozone transactions (no sampling)
    print(f"Using all {len(df_euro)} Eurozone transactions for analysis.")
    print("(Set SAMPLE_SIZE in code if you want to limit for testing)")
    
    # Optional: Uncomment to limit for testing
    # SAMPLE_SIZE = 300
    # if len(df_euro) > SAMPLE_SIZE:
    #     df_sampled = df_euro.sample(n=SAMPLE_SIZE, random_state=42)
    #     print(f"Sampled {SAMPLE_SIZE} transactions for testing.")
    #     return df_sampled
        
    return df_euro

def extract_deal_rationale(transaction_comment):
    """Extract deal rationale categories from transaction comments."""
    if pd.isna(transaction_comment):
        return {'operational': 0, 'financial': 0, 'regulatory': 0, 'technology': 0, 'market_expansion': 0}
    
    text = str(transaction_comment).lower()
    
    # Keywords for each category
    operational_keywords = ['economies of scale', 'efficiency', 'synerg', 'integration', 'cost reduction', 'operational']
    financial_keywords = ['diversif', 'capital', 'debt', 'financing', 'leverage', 'valuation', 'ebitda', 'revenue']
    regulatory_keywords = ['regulat', 'compliance', 'license', 'approval', 'authority', 'patent']
    technology_keywords = ['technology', 'innovation', 'r&d', 'digital', 'platform', 'software', 'tech']
    market_keywords = ['market', 'expansion', 'geographic', 'customer', 'portfolio', 'presence']
    
    rationale = {
        'operational': sum(1 for kw in operational_keywords if kw in text),
        'financial': sum(1 for kw in financial_keywords if kw in text),
        'regulatory': sum(1 for kw in regulatory_keywords if kw in text),
        'technology': sum(1 for kw in technology_keywords if kw in text),
        'market_expansion': sum(1 for kw in market_keywords if kw in text)
    }
    
    return rationale

def extract_financial_metrics_from_text(transaction_comment):
    """Extract financial metrics mentioned in transaction comments."""
    if pd.isna(transaction_comment):
        return {'mentioned_revenue': None, 'mentioned_ebitda': None, 'has_valuation_ratio': 0}
    
    text = str(transaction_comment)
    
    # Extract revenue mentions (€X million, $X million)
    revenue_pattern = r'revenue[s]?\\s+of\\s+[€$£]?([0-9,]+(?:\\.[0-9]+)?)\\s*million'
    revenue_match = re.search(revenue_pattern, text, re.IGNORECASE)
    mentioned_revenue = float(revenue_match.group(1).replace(',', '')) if revenue_match else None
    
    # Extract EBITDA mentions
    ebitda_pattern = r'ebitda\\s+of\\s+[€$£]?([0-9,]+(?:\\.[0-9]+)?)\\s*million'
    ebitda_match = re.search(ebitda_pattern, text, re.IGNORECASE)
    mentioned_ebitda = float(ebitda_match.group(1).replace(',', '')) if ebitda_match else None
    
    # Check if valuation ratios mentioned
    has_valuation = 1 if any(term in text.lower() for term in ['p/e', 'price-to-earnings', 'ev/ebitda', 'valuation']) else 0
    
    return {
        'mentioned_revenue': mentioned_revenue,
        'mentioned_ebitda': mentioned_ebitda,
        'has_valuation_ratio': has_valuation
    }

def extract_key_entities(transaction_comment):
    """Extract key financial entities and advisors."""
    if pd.isna(transaction_comment):
        return {'has_advisor': 0, 'num_advisors': 0}
    
    text = str(transaction_comment)
    
    # Count advisor mentions
    advisor_keywords = ['advisor', 'adviser', 'counsel', 'acted as', 'llp', 'bank']
    has_advisor = 1 if any(kw in text.lower() for kw in advisor_keywords) else 0
    
    # Count number of advisor mentions (rough estimate)
    num_advisors = len(re.findall(r'acted as.*?advisor|acted as.*?counsel', text, re.IGNORECASE))
    
    return {'has_advisor': has_advisor, 'num_advisors': num_advisors}

def fetch_financial_data(ticker):
    """Fetches basic financial info using yfinance."""
    if not ticker or pd.isna(ticker):
        return None
    
    # Tickers in dataset might need adjustment (e.g. adding suffix like .PA for Paris)
    # For now, try as is.
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant fields (handle missing keys gracefully)
        data = {
            'Market Cap': info.get('marketCap'),
            'Enterprise Value': info.get('enterpriseValue'),
            'Trailing PE': info.get('trailingPE'),
            'Forward PE': info.get('forwardPE'),
            'Price to Book': info.get('priceToBook'),
            'Profit Margins': info.get('profitMargins'),
            'Revenue': info.get('totalRevenue'),
            'EBITDA': info.get('ebitda'),
            'Sector': info.get('sector'),
            'Industry': info.get('industry')
        }
        return data
    except Exception as e:
        # print(f"Failed to fetch data for {ticker}: {e}")
        return None

def enrich_data(df):
    """Enriches dataframe with financial data and extracted insights."""
    print("Enriching data with financial metrics and text analysis...")
    
    # Initialize columns
    financial_cols = ['Market Cap', 'Enterprise Value', 'Trailing PE', 
                      'Forward PE', 'Price to Book', 'Profit Margins', 
                      'Revenue', 'EBITDA', 'Sector', 'Industry']
    
    for col in financial_cols:
        df[col] = np.nan
    
    # Extract deal rationale and insights from transaction comments
    print("Analyzing transaction descriptions...")
    rationale_data = df['Transaction Comments'].apply(extract_deal_rationale)
    for key in ['operational', 'financial', 'regulatory', 
                'technology', 'market_expansion']:
        df[f'rationale_{key}'] = rationale_data.apply(lambda x: x[key])
    
    # Extract financial metrics from text
    financial_text_data = df['Transaction Comments'].apply(
        extract_financial_metrics_from_text)
    df['mentioned_revenue'] = financial_text_data.apply(
        lambda x: x['mentioned_revenue'])
    df['mentioned_ebitda'] = financial_text_data.apply(
        lambda x: x['mentioned_ebitda'])
    df['has_valuation_ratio'] = financial_text_data.apply(
        lambda x: x['has_valuation_ratio'])
    
    # Extract advisor information
    entity_data = df['Transaction Comments'].apply(extract_key_entities)
    df['has_advisor'] = entity_data.apply(lambda x: x['has_advisor'])
    df['num_advisors'] = entity_data.apply(lambda x: x['num_advisors'])
        
    # Iterate and fetch financial data from Yahoo Finance
    # Note: Many companies won't have ticker data
    
    print("Fetching market data for publicly traded companies...")
    count = 0
    successful = 0
    
    for index, row in df.iterrows():
        ticker = row['Exchange:Ticker']
        if pd.notna(ticker) and ticker != '-':
            # Parse ticker format: "EXCHANGE:TICKER"
            ticker_parts = str(ticker).split(':')
            if len(ticker_parts) > 1:
                ticker_str = ticker_parts[1]
            else:
                ticker_str = ticker_parts[0]
            
            fin_data = fetch_financial_data(ticker_str)
            if fin_data:
                for key, value in fin_data.items():
                    df.at[index, key] = value
                successful += 1
            
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} tickers ({successful} successful)...")
            
            # Sleep to avoid rate limits
            time.sleep(0.3)
            
    return df

if __name__ == "__main__":
    file_path = "Project/MA Transactions Over 50M.xlsx"
    
    try:
        df = load_and_filter_data(file_path)
        
        # Save intermediate result
        df.to_csv("eurozone_transactions.csv", index=False)
        print("Saved filtered data to eurozone_transactions.csv")
        
        # Enrich
        df_enriched = enrich_data(df)
        df_enriched.to_csv("eurozone_transactions_enriched.csv", index=False)
        print("Saved enriched data to eurozone_transactions_enriched.csv")
        
    except Exception as e:
        print(f"An error occurred: {e}")
