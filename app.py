import streamlit as st
import twitter_api
import predict_live_tweets
import trading_algorithm
import numpy as np
import pandas as pd
import torch
from torch import nn
import time
from transformers import BertModel, AdamW
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=False)
    self.dropout = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.dropout(output)
    return self.out(output)


sp500_stocks = ["$A", "$AAL", "$AAP", "$AAPL", "$ABBV", "$ABC", "$ABMD", "$ABT", "$ACN", "$ADBE", "$ADI", "$ADM", "$ADP", "$ADSK", "$AEE", "$AEP", "$AES", "$AFL", "$AIG", "$AIZ", "$AJG", "$AKAM", "$ALB", "$ALGN", "$ALK", "$ALL", "$ALLE", "$AMAT", "$AMCR", "$AMD", "$AME", "$AMGN", "$AMP", "$AMT", "$AMZN", "$ANET", "$ANSS", "$ANTM", "$AON", "$AOS", "$APA", "$APD", "$APH", "$APTV", "$ARE", "$ATO", "$ATVI", "$AVB", "$AVGO", "$AVY", "$AWK", "$AXP", "$AZO", "$BA", "$BAC", "$BAX", "$BBWI", "$BBY", "$BDX", "$BEN", "$BF.B", "$BIIB", "$BIO", "$BK", "$BKNG", "$BKR", "$BLK", "$BLL", "$BMY", "$BR", "$BRK.B", "$BRO", "$BSX", "$BWA", "$BXP", "$C", "$CAG", "$CAH", "$CARR", "$CAT", "$CB", "$CBOE", "$CBRE", "$CCI", "$CCL", "$CDAY", "$CDNS", "$CDW", "$CE", "$CEG", "$CERN", "$CF", "$CFG", "$CHD", "$CHRW", "$CHTR", "$CI", "$CINF", "$CL", "$CLX", "$CMA", "$CMCSA", "$CME", "$CMG", "$CMI", "$CMS", "$CNC", "$CNP", "$COF", "$COO", "$COP", "$COST", "$CPB", "$CPRT", "$CPT", "$CRL", "$CRM", "$CSCO", "$CSX", "$CTAS", "$CTLT", "$CTRA", "$CTSH", "$CTVA", "$CTXS", "$CVS", "$CVX", "$CZR", "$D", "$DAL", "$DD", "$DE", "$DFS", "$DG", "$DGX", "$DHI", "$DHR", "$DIS", "$DISH", "$DLR", "$DLTR", "$DOV", "$DOW", "$DPZ", "$DRE", "$DRI", "$DTE", "$DUK", "$DVA", "$DVN", "$DXC", "$DXCM", "$EA", "$EBAY", "$ECL", "$ED", "$EFX", "$EIX", "$EL", "$EMN", "$EMR", "$ENPH", "$EOG", "$EPAM", "$EQIX", "$EQR", "$ES", "$ESS", "$ETN", "$ETR", "$ETSY", "$EVRG", "$EW", "$EXC", "$EXPD", "$EXPE", "$EXR", "$F", "$FANG", "$FAST", "$FB", "$FBHS", "$FCX", "$FDS", "$FDX", "$FE", "$FFIV", "$FIS", "$FISV", "$FITB", "$FLT", "$FMC", "$FOX", "$FOXA", "$FRC", "$FRT", "$FTNT", "$FTV", "$GD", "$GE", "$GILD", "$GIS", "$GL", "$GLW", "$GM", "$GNRC", "$GOOG", "$GOOGL", "$GPC", "$GPN", "$GRMN", "$GS", "$GWW", "$HAL", "$HAS", "$HBAN", "$HCA", "$HD", "$HES", "$HIG", "$HII", "$HLT", "$HOLX", "$HON", "$HPE", "$HPQ", "$HRL", "$HSIC", "$HST", "$HSY", "$HUM", "$HWM", "$IBM", "$ICE", "$IDXX", "$IEX", "$IFF", "$ILMN", "$INCY", "$INTC", "$INTU", "$IP", "$IPG", "$IPGP", "$IQV", "$IR", "$IRM", "$ISRG", "$IT", "$ITW", "$IVZ",
                "$J", "$JBHT", "$JCI", "$JKHY", "$JNJ", "$JNPR", "$JPM", "$K", "$KEY", "$KEYS", "$KHC", "$KIM", "$KLAC", "$KMB", "$KMI", "$KMX", "$KO", "$KR", "$L", "$LDOS", "$LEN", "$LH", "$LHX", "$LIN", "$LKQ", "$LLY", "$LMT", "$LNC", "$LNT", "$LOW", "$LRCX", "$LUMN", "$LUV", "$LVS", "$LW", "$LYB", "$LYV", "$MA", "$MAA", "$MAR", "$MAS", "$MCD", "$MCHP", "$MCK", "$MCO", "$MDLZ", "$MDT", "$MET", "$MGM", "$MHK", "$MKC", "$MKTX", "$MLM", "$MMC", "$MMM", "$MNST", "$MO", "$MOH", "$MOS", "$MPC", "$MPWR", "$MRK", "$MRNA", "$MRO", "$MS", "$MSCI", "$MSFT", "$MSI", "$MTB", "$MTCH", "$MTD", "$MU", "$NCLH", "$NDAQ", "$NDSN", "$NEE", "$NEM", "$NFLX", "$NI", "$NKE", "$NLOK", "$NLSN", "$NOC", "$NOW", "$NRG", "$NSC", "$NTAP", "$NTRS", "$NUE", "$NVDA", "$NVR", "$NWL", "$NWS", "$NWSA", "$NXPI", "$O", "$ODFL", "$OGN", "$OKE", "$OMC", "$ORCL", "$ORLY", "$OTIS", "$OXY", "$PARA", "$PAYC", "$PAYX", "$PCAR", "$PEAK", "$PEG", "$PENN", "$PEP", "$PFE", "$PFG", "$PG", "$PGR", "$PH", "$PHM", "$PKG", "$PKI", "$PLD", "$PM", "$PNC", "$PNR", "$PNW", "$POOL", "$PPG", "$PPL", "$PRU", "$PSA", "$PSX", "$PTC", "$PVH", "$PWR", "$PXD", "$PYPL", "$QCOM", "$QRVO", "$RCL", "$RE", "$REG", "$REGN", "$RF", "$RHI", "$RJF", "$RL", "$RMD", "$ROK", "$ROL", "$ROP", "$ROST", "$RSG", "$RTX", "$SBAC", "$SBNY", "$SBUX", "$SCHW", "$SEDG", "$SEE", "$SHW", "$SIVB", "$SJM", "$SLB", "$SNA", "$SNPS", "$SO", "$SPG", "$SPGI", "$SRE", "$STE", "$STT", "$STX", "$STZ", "$SWK", "$SWKS", "$SYF", "$SYK", "$SYY", "$T", "$TAP", "$TDG", "$TDY", "$TECH", "$TEL", "$TER", "$TFC", "$TFX", "$TGT", "$TJX", "$TMO", "$TMUS", "$TPR", "$TRMB", "$TROW", "$TRV", "$TSCO", "$TSLA", "$TSN", "$TT", "$TTWO", "$TWTR", "$TXN", "$TXT", "$TYL", "$UA", "$UAA", "$UAL", "$UDR", "$UHS", "$ULTA", "$UNH", "$UNP", "$UPS", "$URI", "$USB", "$V", "$VFC", "$VLO", "$VMC", "$VNO", "$VRSK", "$VRSN", "$VRTX", "$VTR", "$VTRS", "$VZ", "$WAB", "$WAT", "$WBA", "$WBD", "$WDC", "$WEC", "$WELL", "$WFC", "$WHR", "$WM", "$WMB", "$WMT", "$WRB", "$WRK", "$WST", "$WTW", "$WY", "$WYNN", "$XEL", "$XOM", "$XRAY", "$XYL", "$YUM", "$ZBH", "$ZBRA", "$ZION", "$ZTS"]
class_names = ['bearish', 'bullish']
model = SentimentClassifier(len(class_names))
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
device = torch.device('cpu')
checkpoint = torch.load("twitter_sentiment_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
equities = []
datetimes = []
    
def trade(portfolio):
    '''
    https://developer.twitter.com/en/docs/twitter-api/rate-limits

    A maximum of 450 recent search requests for tweets are allowed every 15 minutes.
    Making more requests than this will result in an error.
    This works out to one request made every 2 seconds.

    To be safe, since time.sleep is approximate - pause for 5 seconds after tweet queries
    and 3 seconds after stock order queries, which is more than enough wait time.

    This means that a particular ticker, such as $AAPL, will be checked on every time.sleep(#) * len(hash_tag_list) seconds.
    '''
    global equities, datetimes

    twitter_client = twitter_api.TwitterClient()
    tweet_preprocessor = twitter_api.TweetPreprocessor()
    api = twitter_client.get_twitter_client_api()
    latest_id = None

    hash_tag_dict = dict.fromkeys(portfolio, None)
    placeholder = st.empty()

    while True:
        with placeholder.container():
            if len(equities) > 1:
                chart_data = pd.DataFrame(equities, columns=['Equity'])
                st.line_chart(chart_data)
            # if len(equities) > 1:
            #     fig, ax = plt.subplots(figsize=(6, 0.5))
            #     ax.plot(datetimes, equities)
            #     ax.set(title='Portfolio Performance',
            #            ylabel='Equity',
            #            xlabel='Date-Time')
            #     st.pyplot(fig)

            for ticker in hash_tag_dict.keys():
                tweets = api.search(
                    q=ticker, lang='en', result_type='recent', since_id=hash_tag_dict[ticker])
                df, latest_id = tweet_preprocessor.tweets_to_data_frame(tweets)
                # update the value of the current ticker to the latest tweet id
                hash_tag_dict[ticker] = latest_id
                polarity, trade_bool, df = predict_live_tweets.predict_tweets(
                    model, df, risk_level=RISK_LEVEL)

                # ### useful print codes to illustrate what is happening:
                print(df[['tweet', 'prediction', 'id']].head(3))
                st.write()
                equity, timestamp, msg = trading_algorithm.alpaca_trader(ticker, polarity)
                equities.append(equity)
                datetimes.append(str(timestamp).split('.')[0])
                st.write(msg)
                st.write(timestamp)
                st.write(f'Portfolio Balance: ${float(equity):0,.0f}')
                
                if msg != 'No orders were made because the stock market is currently closed for trading. \n':
                    st.write(f"Searching for recent tweets about {ticker}")

                    if trade_bool:
                        if polarity > 0:
                            st.write(
                                f"Positive sentiment polarity score of {polarity}. Preparing to buy {ticker} stock.")
                        else:
                            st.write(
                                f"Negative sentiment polarity score of {polarity}. Avoiding or preparing to sell {ticker} stock.")
                        st.write()
                    else:
                        st.write(
                            f"Neutral sentiment polarity score. Hold off on trading {ticker} stock.")
                time.sleep(5)  # seconds
                st.write()
                st.write('--------------------')
                st.write()
        placeholder.empty()




# main()
st.title("Stock Trader using Tweet Sentiment")
st.write("##### A BERT Sentiment Classifier by Eric Zacharia")
portfolio = st.multiselect('Build your portfolio', sp500_stocks)
st.write("Choose how risky you want your bot's trades to be.")
RISK_LEVEL = st.slider(
    'Most Conservative ....................................................................................................................................................... Most Risky', 0, 100, 20) / 100

if st.button('Start Trading') and portfolio:
    if st.button('Stop Trading'):
        st.write("Stopping...")
    st.write('Running...')
    trade(portfolio)

elif not portfolio:
    st.write('Add stocks to your portfolio to begin trading.')


# model = cnn(input_shape=(288, 432, 4), classes=9)
# model.load_weights("CNNModelWeights.h5")

# if file is not None:
#     convert_mp3_to_wav(file)
#     extract_relevant("music_file.wav", 40, 50)
#     create_melspectrogram("extracted.wav")
#     image_data = load_img('melspectrogram.png',
#                           color_mode='rgba', target_size=(288, 432))

#     filename = download_image()
#     st.write(f"### {filename}")
#     st.audio(file, "audio/mp3")

    # class_label, prediction = predict(image_data, model)
    # prediction = prediction.reshape((9,))
    # color_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # my_cmap = cm.get_cmap('jet')
    # my_norm = Normalize(vmin=0, vmax=9)

    # st.write(f"### Portfolio Performance Graph")
    # plt.plot(datetimes, equity)
    # plt.title('Equity vs Time')
    # plt.xlabel('Timestamps')
    # plt.ylabel('Equity')
    # plt.show()


    # fig, ax = plt.subplots(figsize=(6, 4.5))
    # ax.bar(x=class_labels, height=prediction,
    #        color=my_cmap(my_norm(color_data)))
    # plt.xticks(rotation=45)
    # ax.set_title(
    #     "Probability Distribution Of The Given Song Over Different Genres")
    # st.pyplot(fig)
    # st.image("melspectrogram.png", use_column_width=True)
