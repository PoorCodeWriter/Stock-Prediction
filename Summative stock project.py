import tkinter as tk
from tkinter import Label, Listbox, Toplevel
import numpy as np
import statsmodels.api as sm
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import webbrowser
import requests
import pandas as pd


def read_tickers(file_path):
    global tickerSymbol
    data = pd.read_csv(file_path, sep="\t")  # assuming the data is tab-separated

    print(data.columns)

    name = input("Enter the name of the company: ")

    matching = []

    for index, row in data.iterrows():
        if name.lower() in row['Name'].lower():
            matching.append((row['Symbol'], row['Name']))

    if matching:
        print(f"Symbols for companies matching the name '{name}':")
        for symbol, name in matching:
            print(f"Symbol: {symbol}, Name: {name}")

    else:
        print(f"No companies found matching the name '{name}'.")
        read_tickers(file_path)

    tickerSymbol = input("Enter Ticker: ")
    matching_names = []

    for index, row in data.iterrows():
        if tickerSymbol.lower() == row['Symbol'].lower():
            matching_names.append(row['Name'])

    if matching_names:
        print(f"Name for ticker '{tickerSymbol}':")
        for name in matching_names:
            print(f"Name: {name}")
    else:
        print(f"No name found for ticker '{tickerSymbol}'.")
        read_tickers(file_path)


read_tickers("C:/Users/Clinton.Luk/Documents/tsx_companies.txt")
fig = go.Figure()
lambda_reg = 0.2
NEWS_API_KEY = "52679ebab7b14c30914a340a860b9c5d"
popup = None


def calculate_probability(data):
    total = len(data)
    up_days = len(data[data['Close'] > data['Open']])
    down_days = len(data[data['Close'] < data['Open']])
    up = up_days / total
    down = down_days / total
    return up, down


def get_news_data(company_name):
    base_url = "https://newsapi.org/v2/everything?"
    complete_url = base_url + "apikey=" + NEWS_API_KEY + "&q=" + company_name
    response = requests.get(complete_url)
    data = response.json()

    if data["status"] == "ok":
        return data["articles"]
    else:
        return None


def open_url(url):
    webbrowser.open(url)


def close_popup_and_quit():
    global popup
    if popup is not None:
        popup.destroy()
    app.quit()


def create_popup(stock_symbol):
    global news
    Pop = tk.Toplevel(app)
    Pop.geometry("500x400")
    Pop.title("Stock Market Prediction - " + stock_symbol)
    # Create a listbox to display news articles
    news = tk.Listbox(Pop, width=70)
    news.pack(fill=tk.BOTH, expand=True)

    news_data = get_news_data(stock_symbol)
    if news_data:
        news.delete(0, tk.END)
        for article in news_data[:14]:
            news.insert(tk.END, f"{article['title']} ({article['source']['name']})")
    else:
        news.delete(0, tk.END)
        news.insert(tk.END, "No news found for the specified ticker symbol")
    # Define labels as global variables
    global current_price, highest_price, lowest_price, accuracy, probability_up, \
        probability_down, buy_in_price

    current_price = tk.Label(Pop)
    highest_price = tk.Label(Pop)
    lowest_price = tk.Label(Pop)
    accuracy = tk.Label(Pop)
    probability_up = tk.Label(Pop)
    probability_down = tk.Label(Pop)
    buy_in_price = tk.Label(Pop)

    # Pack the labels
    current_price.pack()
    highest_price.pack()
    lowest_price.pack()
    accuracy.pack()
    probability_up.pack()
    probability_down.pack()
    buy_in_price.pack()

    # Add a button to close the popup
    Close = tk.Button(Pop, text="Close", command=close_popup_and_quit)
    Close.pack()

    # Refresh the data immediately
    refresh_stock_data()


def analysis(data):
    # Calculate the 50-day and 200-day moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Generate trading signals based on the moving average crossover
    data['Signal'] = np.where(data['MA50'] > data['MA200'], 1, -1)

    # Calculate the points of crossover
    data['Crossover'] = np.where(data['Signal'].diff() != 0, data['Close'], np.nan)

    # Identify potential entry and exit points
    entry_points = data[data['Crossover'].notnull() & (data['Signal'] == 1)]
    exit_points = data[data['Crossover'].notnull() & (data['Signal'] == -1)]

    return entry_points, exit_points


def refresh_stock_data():
    # Get data on the stock
    tickerData = yf.Ticker(tickerSymbol)
    # get the price history of the stock
    tickerDf = tickerData.history(period='1d', start='2023-5-5', end='2024-5-1')

    # Create a new column 'Price_Up' that contains 1 if the closing price of tomorrow is greater than the closing
    # price of today
    tickerDf['Price_Up'] = np.where(tickerDf['Close'].shift(-1) > tickerDf['Close'], 1, 0)

    # Drop unnecessary columns
    tickerDf = tickerDf[['Open', 'High', 'Low', 'Close', 'Price_Up']]

    # Define features (X) and target (y)
    X = tickerDf.drop('Price_Up', axis=1)
    y = tickerDf['Price_Up']

    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['Close'], mode='lines'))

    # Split the data into training and testing sets
    XT, X, YT, Y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add a constant column to XT
    XT = sm.add_constant(XT)

    # Create and train the logistic regression model with L2 regularization
    logit_model = sm.Logit(YT, XT)
    result = logit_model.fit_regularized(alpha=lambda_reg)

    # Test the model
    X = sm.add_constant(X)
    predictions = result.predict(X)

    # Calculate the AC of the model
    AC = accuracy_score(Y, np.round(predictions))
    # highest price
    HP = tickerDf['High'].max()
    # lowest price
    LP = tickerDf['Low'].min()
    # current price
    CP = tickerDf['Close'][-1]
    # probability of the stock going up or down
    PU, PD = calculate_probability(tickerDf)
    # Buy in price
    BIP = tickerDf['Close'][-2] if tickerDf['Price_Up'][-1] == 1 else tickerDf['Close'][-3]
    # Update the labels in the popup with the latest information
    current_price.config(text=f"Current price: {CP:.2f}")
    highest_price.config(text=f"Highest price ever: {HP:.2f}")
    lowest_price.config(text=f"Lowest price ever: {LP:.2f}")
    accuracy.config(text=f"Accuracy: {AC:.2%}")
    probability_up.config(text=f"Probability of the stock going up: {PU:.2%}")
    probability_down.config(text=f"Probability of the stock going down: {PD:.2%}")
    buy_in_price.config(text=f"Ideal buy-in price: {BIP:.2f}")

    # Perform technical analysis to identify potential entry and exit points
    EP, EXP = analysis(tickerDf)

    # Plot entry points on the stock history graph
    fig.add_trace(go.Scatter(x=EP.index, y=EP['Crossover'], mode='markers', marker=dict(color='green'),
                             name='Entry Point'))

    # Plot exit points on the stock history graph
    fig.add_trace(go.Scatter(x=EXP.index, y=EXP['Crossover'], mode='markers', marker=dict(color='red'),
                             name='Exit Point'))

    # Update the stock history graph
    fig.update_layout(title='Stock History', xaxis_title='Date', yaxis_title='Price')
    fig.show()

    # refresh
    app.after(20000, refresh_stock_data)


# Create the main application window
app = tk.Tk()
app.title(tickerSymbol)
app.withdraw()  # Hide the main window

# Call create_popup to initialize the popup
create_popup(tickerSymbol)

# Start the main application loop
app.mainloop()
