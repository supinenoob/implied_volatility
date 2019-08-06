
#------------------------Packages------------------------#

import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import pymc3 as pm
### I have 252 observations for HV, so 0 to 251 ###
### I have 253 observations for IV, so 0 to 252 ###


#-------------------------Option Pricing---------------------------#

    ######## Black-Scholes Model########

''' Call Option Premium C = SN(d1) - Xe^(-rt) N(d2)
     Put Option Premium P = Xe^(-rT) N(-d2) - S0 N(-d1)'''

def BS_price(strategy, sp, x, r, sd, t):
    d1 = (np.log(sp/x) + (r + (sd**2*0.5)*t)) / sd*np.sqrt(t)
    #print("D1 = ", d1)
    d2 = d1 - sd*np.sqrt(t)
    if strategy == 'c':
        price = sp * si.norm.cdf(d1,0,1) - x * np.exp(-r*t) * si.norm.cdf(d2,0,1)
        return price
    elif strategy == 'p':
        price = x * np.exp(-r*t) * si.norm.cdf(-d2,0,1) - sp * si.norm.cdf(-d1,0,1)
        return price
    else:
        print("No Such Option Strategy; Enter 'c' for Call or 'p' for Put")
        exit()

    ######## Calculating Historical Volatility (A.K.A standard deviation of the periodic daily return) #######
    #'n' Max is 252 days#

def hist_vol(ticker, n):
    file = []
    if ticker == 'GOOG':
        file = open('/Users/TristanTaylor/Documents/Master_Project/HV/GOOG_HV.csv', 'r')
    elif ticker == 'AMZN':
        file = open('/Users/TristanTaylor/Documents/Master_Project/HV/AMZN_HV.csv', 'r')
    elif ticker == 'AAPL':
        file = open('/Users/TristanTaylor/Documents/Master_Project/HV/AAPL_HV.csv', 'r')
    else:
        print("Enter 'GOOG', 'AMZN', or 'AAPL' for ticker")
        exit()
    s = []
    for lines in file:
        s = np.append(s, float(lines))
    #print(s)
    r = []
    for i in range(n):
        r.append(np.log(s[i]/s[i+1]))
        #print(r)

    a = []
    r_mean = np.mean(r)
    #print('Mean: ', r_mean)
    diff_square = [(r[j] - r_mean) ** 2 for j in range(0, len(r))]
    #print(diff_square)
    std = np.sqrt(sum(diff_square) * (1.0 / (n)))
    hist = std * np.sqrt(n+1)
    return hist, std


####GOOG on 06/24/19: adj c.price=1115.52, strike=1115.00, 10yr. Treasury=0.02016, time=88days, target=45.30, "call"

#OTM Call & Put
####GOOG on 06/27/19: adj c.price=1076.01, strike=1077.50, 10yr. Treasury=0.02016, time=29days, target=35.55, "call"
####GOOG on 06/27/19: adj c.price=1076.01, strike=1075.00, 10yr. Treasury=0.02016, time=29days, target=33.75, "put"

####AMZN on 06/27/19: adj c.price=1904.28, strike=1905.00, 10yr. Treasury=0.02016, time=29days, target=66.00, "call"
####AMZN on 06/27/19: adj c.price=1904.28, strike=1902.50, 10yr. Treasury=0.02016, time=29days, target=61.48, "put"

####AAPL on 6/27/19: adj c.price=199.74, strike=200.00, 10yr. Treasury=0.02016, time=29days, target=5.07, "call"
####AAPL on 6/27/19: adj c.price=199.74, strike=197.50, 10yr. Treasury=0.02016, time=29days, target=4.53, "put"

#25 Delta Call & Put
####GOOG on 06/27/19: adj c.price=1076.01, strike=1095.00, 10yr. Treasury=0.02016, time=29days, target=27.05, "call"
####GOOG on 06/27/19: adj c.price=1076.01, strike=1057.50, 10yr. Treasury=0.02016, time=29days, target=24.65, "put"

####AMZN on 06/27/19: adj c.price=1904.28, strike=1922.50, 10yr. Treasury=0.02016, time=29days, target=56.98, "call"
####AMZN on 06/27/19: adj c.price=1904.28, strike=1885.00, 10yr. Treasury=0.02016, time=29days, target=53.58, "put"

####AAPL on 06/27/19: adj c.price=199.74, strike=217.50, 10yr. Treasury=0.02016, time=29days, target=0.6, "call"
####AAPL on 06/27/19: adj c.price=199.74, strike=180.00, 10yr. Treasury=0.02016, time=29days, target=0.85, "put"

    ######## Calculating Vega ########

def vega(sp, x, r, sd, t):
    d1 = (np.log(sp/x) + (r + (sd**2/2)*t)) / sd*np.sqrt(t)
    return sp * np.sqrt(t) * si.norm.pdf(d1,0,1)

    ######### Calculating Implied Volatility########

def imp_vol(target, strategy, sp, x, r, sd, t):
    precision = 0.0000001

    price = BS_price(strategy, sp, x, r, sd, t)
    veg = vega(sp, x, r, sd, t)
    diff = target - price
    new_sd = sd + diff / (veg)

    if abs(diff) > precision:
        new_sd = imp_vol(target, strategy, sp, x, r, new_sd, t)
    else:
        new_sd = sd + diff/vega(sp, x, r, new_sd, t)
        #print('New SD = ', new_sd)

    return new_sd


def iv_range_formula(sp, sd, t):
    upper = sp + sp*sd*np.sqrt(t)
    lower = sp - sp*sd*np.sqrt(t)
    return upper, lower

def iv_percent(ticker, strategy, n, sd):
    file = []
    if ticker == "GOOG":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_put_IV2.csv', 'r')
    elif ticker == "AMZN":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_put_IV2.csv', 'r')
    elif ticker == "AAPL":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_put_IV2.csv', 'r')
    else:
        exit()
    s = []
    for lines in file:
        s = np.append(s, float(lines))
    # print(s)
    r = []
    for i in range(n):
        r.append(s[i])
        # print(r)
    b = 0
    for i in range(len(r)):
        if r[i] < sd*100:
            b += 1
        else:
            pass
    p = b/252
    return p

#----------------------Bayesian Inference-------------------------#

    ######## Plotting Implied Volatility for GOOG, AAPL, & AMZN ########

def data_iv(ticker, strategy, n):
    file = []
    if ticker == "GOOG":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_put_IV2.csv', 'r')
    elif ticker == "AMZN":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_put_IV2.csv', 'r')
    elif ticker == "AAPL":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_put_IV2.csv', 'r')
    else:
        exit()
    s = []
    for lines in file:
        s = np.append(s, float(lines))
    # print(s)
    r = []
    for i in range(n):
        r.append(s[i])
        # print(r)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(r, label='Implied Volatility Plot')
    ax.set(xlabel='Time (Days)', ylabel='25 Trading Days IV')
    ax.legend()
    plt.show()
    r_mean = np.mean(r)
    #print('Mean: ', r_mean)
    diff_square = [(r[j] - r_mean) ** 2 for j in range(0, len(r))]
    # print(diff_square)
    var = (np.sum(diff_square) / (n))
    # std = np.sqrt(np.sum(diff_square)/(n))
    #print('Variance: ', var)
    # print(std)
    # hist = std * np.sqrt(n+1)
    return r_mean, var

def data_iv_update(ticker, strategy, n):
    file = []
    if ticker == "GOOG":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_call_IV.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_put_IV.csv', 'r')
    elif ticker == "AMZN":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_call_IV.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_put_IV.csv', 'r')
    elif ticker == "AAPL":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_call_IV.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_put_IV.csv', 'r')
    else:
        exit()
    s = []
    for lines in file:
        s = np.append(s, float(lines))
    # print(s)
    r = []
    for i in range(n):
        r.append(s[i])
        # print(r)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(r, label='Implied Volatility Plot')
    ax.set(xlabel='Time (Days)', ylabel='25 Trading Days IV')
    ax.legend()
    plt.show()
    r_mean = np.mean(r)
    # print('Mean: ', r_mean)
    diff_square = [(r[j] - r_mean) ** 2 for j in range(0, len(r))]
    # print(diff_square)
    var = (np.sum(diff_square) / (n))
    # std = np.sqrt(np.sum(diff_square)/(n))
    # print('Variance: ', var)
    # print(std)
    # hist = std * np.sqrt(n+1)
    return r_mean, var


    ######## Methods of Moments Estimation of Gamma Parameters ########

def parameters(mean, var):
    shape = mean ** 2 / var
    #print('Shape: ', shape)
    scale = var / mean
    #print('Scale: ', scale)
    return shape, scale

def parameters_inv(mean, var):
    shape = mean ** 2 / var + 2
    scale = mean * (mean ** 2 / var + 1)
    return shape, scale

    ######## QQPlot: See if Data Follows a Distribution ########

def qqplot(ticker, strategy, n, mean, var, shape, scale, inv_shape, inv_scale):
    file = []
    if ticker == "GOOG":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_put_IV2.csv', 'r')
    elif ticker == "AMZN":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_put_IV2.csv', 'r')
    elif ticker == "AAPL":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_call_IV2.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_put_IV2.csv', 'r')
    else:
        exit()
    s = []
    for lines in file:
        s = np.append(s, float(lines))
    #print(s)
    r = []
    for i in range(n):
        r.append(s[i])
    #print(r)

    fig = plt.figure()
    ax_1 = fig.add_subplot(221)
    ax_2 = fig.add_subplot(222)
    ax_3 = fig.add_subplot(223)
    res_g = si.probplot(r, dist=si.gamma, sparams=(shape, scale), plot=ax_1)
    res_n = si.probplot(r, dist=si.norm, sparams=(mean, np.sqrt(var)), plot=ax_2)
    res_i = si.probplot(r, dist=si.invgamma, sparams=(inv_shape, inv_scale), plot=ax_3)
    ax_1.set_title("Probplot on Gamma Distribution")
    ax_2.set_title("Probplot on Normal Distribution")
    ax_3.set_title("Probplot on Inverse Gamma Distribution")
    plt.show()
    return r


def qqplot_update(ticker, strategy, n, mean, var, shape, scale, inv_shape, inv_scale):
    file = []
    if ticker == "GOOG":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_call_IV.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/GOOG_put_IV.csv', 'r')
    elif ticker == "AMZN":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_call_IV.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AMZN_put_IV.csv', 'r')
    elif ticker == "AAPL":
        if strategy == 'c':
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_call_IV.csv', 'r')
        else:
            file = open('/Users/TristanTaylor/Documents/Master_Project/IV/AAPL_put_IV.csv', 'r')
    else:
        exit()
    s = []
    for lines in file:
        s = np.append(s, float(lines))
    #print(s)
    r = []
    for i in range(n):
        r.append(s[i])
    #print(r)

    fig = plt.figure()
    ax_1 = fig.add_subplot(221)
    ax_2 = fig.add_subplot(222)
    ax_3 = fig.add_subplot(223)
    res_g = si.probplot(r, dist=si.gamma, sparams=(shape, scale), plot=ax_1)
    res_n = si.probplot(r, dist=si.norm, sparams=(mean, np.sqrt(var)), plot=ax_2)
    res_i = si.probplot(r, dist=si.invgamma, sparams=(inv_shape, inv_scale), plot=ax_3)
    ax_1.set_title("Probplot on Gamma Distribution")
    ax_2.set_title("Probplot on Normal Distribution")
    ax_3.set_title("Probplot on Inverse Gamma Distribution")
    plt.show()
    return r


def bayesian_modeling(mean, var, alpha, beta, inv_a, inv_b, iv):
    with pm.Model() as model:
        prior = pm.InverseGamma('bv', inv_a, inv_b)

        likelihood = pm.Gamma('like', alpha, beta, observed=iv)

    with model:
        # step = pm.Metropolis()

        v_trace = pm.sample(10000, tune=1000)
        #print(v_trace['bv'][:])
        trace = v_trace['bv'][:]
        #print(trace1)

    pm.traceplot(v_trace)
    plt.show()

    pm.autocorrplot(v_trace)
    plt.show()

    #s = pm.summary(v_trace)
    #print(s)
    return trace


def update_bayesian_modeling(mean_upd, var_upd, alpha_upd, beta_upd, inv_a_upd, inv_b_upd, iv_upd, strategy, stock_price, strike_price, risk_free, time):
    with pm.Model() as update_model:
        prior = pm.InverseGamma('bv', inv_a_upd, inv_b_upd)

        likelihood = pm.InverseGamma('like', inv_a_upd, inv_b_upd, observed=iv_upd)

    with update_model:
        # step = pm.Metropolis()

        v_trace_update = pm.sample(10000, tune=1000)
        #print(v_trace['bv'][:])
        trace_update = v_trace_update['bv'][:]
        #print(trace)

    pm.traceplot(v_trace_update)
    plt.show()

    pm.autocorrplot(v_trace_update)
    plt.show()

    pm.plot_posterior(v_trace_update[100:], color='#87ceeb', point_estimate='mean')
    plt.show()

    s = pm.summary(v_trace_update).round(2)
    print("\n Summary")
    print(s)

    a = np.random.choice(trace_update, 10000, replace=True)
    ar = []
    for i in range(9999):
        t = a[i] / 100
        ar.append(t)
    #print("Bayesian Volatility Values", ar)

    op = []
    for i in range(9999):
        temp = BS_price(strategy, stock_price, strike_price, risk_free, ar[i], time)
        op.append(temp)
    #print("Bayesian Option Prices", op)

    plt.hist(ar, bins=50)
    plt.title("Volatility")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(op, bins=50)
    plt.title("Option Price")
    plt.ylabel("Frequency")
    plt.show()
    return trace_update


#--------------------------Main Code to Run--------------------------#
def main():
    ######## Getting Input from User ########

    ticker = input("Enter stock ticker symbol: ")

    stock_price = float(input("Enter current stock price: "))

    strike_price = float(input("Enter strike price: "))

    risk_free = float(input("Enter risk-free interest rate: "))

    time = float(input("Enter days to maturity: "))
    time = time/365

    target = float(input("Enter Target Value: "))

    strategy = input("Enter Strategy: ")

    ######## Calculate Implied Volatility & Black-Scholes Option Price w/ Historical Volatility ########

    hist = hist_vol(ticker, 252)

    vol = imp_vol(target, strategy, stock_price, strike_price, risk_free, hist[0], time)

    bs = BS_price(strategy, stock_price, strike_price, risk_free, vol, time)

    percent = iv_percent(ticker, strategy, 252, vol)

    std_dev = iv_range_formula(stock_price, vol, time)

    print("\n Annual Historical Volatility, Periodic Daily Return:")
    print(hist)

    print("\n Option Price with HV:")
    print(BS_price(strategy, stock_price, strike_price, risk_free, hist[0], time))

    print("\n Implied Volatility:")
    print(vol)
    print("\n Option Price with IV:")
    print(bs)
    print("\n IV Percentile")
    print(percent)
    print("\n Expected Price Range at Maturity")
    print(std_dev)


    ######## Calculate Parameters: mean, variance, alpha, & beta ########

    mean, var = data_iv(ticker, strategy, 252)
    mean = float(mean)
    alpha, beta = parameters(mean, var)
    print("\n Gamma Parameters")
    print(alpha, beta)
    inv_a, inv_b = parameters_inv(mean, var)
    print("\n Inverse Gamma Parameters")
    print(inv_a, inv_b)
    iv = qqplot(ticker, strategy, 252, mean, var, alpha, beta, inv_a, inv_b)

    ######## Simple Bayesian Statistics (NUTS) ########

    trace = bayesian_modeling(mean, var, alpha, beta, inv_a, inv_b, iv)

    ######## Define New Parameters: mean, variance, alpha, & beta for Update ########

    mean_update, var_update = np.mean(trace), np.var(trace)
    mean_update = float(mean)
    alpha_update, beta_update = parameters(mean_update, var_update)
    print("\n Gamma Parameters")
    print(alpha_update, beta_update)
    inv_a_update, inv_b_update = parameters_inv(mean_update, var_update)
    print("\n Inverse Gamma Parameters")
    print(inv_a_update, inv_b_update)
    iv_update = qqplot_update(ticker, strategy, 252, mean_update, var_update, alpha_update, beta_update, inv_a_update, inv_b_update)
    new_mean, new_var = data_iv_update(ticker, strategy, 252)
    new_alpha, new_beta = parameters(new_mean, new_var)
    new_inv_a, new_inv_b = parameters_inv(new_mean, new_var)
    ######## Update Posterior ########

    new_trace = update_bayesian_modeling(mean_update, var_update, new_alpha, new_beta, new_inv_a, new_inv_b, iv_update, strategy, stock_price, strike_price, risk_free, time)

    ###CALL###
    # 03/20/18-03/20/19: GOOG mean=23.939, var=33.696, shape=17.008, scale=1.408, bma=23.995
    # 03/26/18-03/26/19: AMZN mean=30.104, var=72.564, shape=12.489, scale=2.410, bma=30.132
    # 03/26/18-03/26/19: AAPL mean=23.539, var=35.819, shape=15.469, scale=1.522, bma=23.463

    ###PUT###
    # 03/20/18-03/20/19: GOOG mean=26.395, var=45.190, shape=15.417, scale=1.712, bma=26.412
    # 03/26/18-03/26/19: AMZN mean=33.377, var=97.999, shape=11.367, scale=2.936, bma=33.376
    # 03/26/18-03/26/19: AAPL mean=26.570, var=57.393, shape=12.300, scale=2.160, bma=26.623

main()
