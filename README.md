# Stock Trend Forecast 
## -based on corporate financial news
**Instruction** This is a project about my 2017 HKU dissertation. For details display you can refer to this [website]().

### Introduction
**Financial news** is always an important indication for the financial market, especially for the **fundamental analysis**, except for statistic materials (including company policies, financial reports and macro-industrial reports). **Technical analysis** would do the 'chart job' and use the stock own data to analyze the market behavior. 
Although each analysis method is important in the stock analysis field, *the human itself is still the only bridge to link two parts together to get the results*. 

This dissertation focus on the corporate financial news with technical stock analysis method to find possible correlations between them and simulate a better **stock trend**.

### Algorithm
**[Text mining](https://en.wikipedia.org/wiki/Text_mining)**: sentiment analysis-python pattern library, lexical analysis-NLTK tokenizer

**[Time series analysis](https://en.wikipedia.org/wiki/Time_series)**: Stationarity Test-AD-Fuller and moving average, ARIMA model

**[Correlation and Regression](https://en.wikipedia.org/wiki/Regression)**: Spearman, Pearson correlation analysis; OLS, Isotonic regression, Ridge regression, Bayes regression

**Stock trend forecast**: ARIMA model integrate with news value attribute

### Code
This project was coded by **Python** using the numpy, pandas, statstools, sklearn, matplotlib library

```python
import numpy
import pandas
import matplotlib
import statstools
```
