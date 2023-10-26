import sys
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd


if len(sys.argv) < 3:
    print("Usage: python3 scrape.py <date> <xslx>")
    exit(1)

desired_date = sys.argv[1]
outpath = sys.argv[2]

url = 'https://discoveratlanta.com/events/all/'
response = rq.get(url)
text = response.content
soup = bs(text, 'html.parser')
articles = soup.find_all('article')

# Walk through the articles, gather titles and urls
titles = []
urls = []
for article in articles:
    dates = article.get('data-eventdates')
    if dates is None:
        continue

    if desired_date in dates:
        heading = article.find('h4', {'class':"listing-title"})
        link = heading.find('a')
        titles.append(link.contents[0])
        urls.append(link.get('href'))

# Create a dataframe
df = pd.DataFrame({'title': titles, 'link':urls})

# Write to an Excel file
writer = pd.ExcelWriter(outpath) 
df.to_excel(writer, sheet_name='Events', index=False)

# Set widths
col_idx = df.columns.get_loc('link')
writer.sheets['Events'].set_column(col_idx, col_idx, 70)
col_idx = df.columns.get_loc('title')
writer.sheets['Events'].set_column(col_idx, col_idx, 40)

# Write it out
writer.save()