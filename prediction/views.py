from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django import forms
from prediction.models import User
from prediction.models import Item
from prediction.models import Hub17
from prediction.models import Hub17_Result
from django.http import *
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import pandas
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import numpy
from sklearn import linear_model
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import r2_score
import warnings
import csv


class UserForm(forms.Form):
    email = forms.CharField(label="email", max_length=50)
    password = forms.CharField(label="password", widget=forms.PasswordInput())


class newUserForm(forms.Form):
    email = forms.CharField(label="email", max_length=50)
    username = forms.CharField(label="username", max_length=50)
    password = forms.CharField(label="password", widget=forms.PasswordInput())


class itemInfoForm(forms.Form):
    SRV_HUB = forms.CharField(label="SRV_HUB", max_length=50)
    BRANCH_CODE = forms.CharField(label="BRANCH_CODE", max_length=50)
    INV_ITEM_ID = forms.CharField(label="INV_ITEM_ID", max_length=50)
    MODEL = forms.CharField(label="MODEL", max_length=50)


class branchInfoForm(forms.Form):
    SRV_HUB = forms.CharField(label="SRV_HUB", max_length=50)
    BRANCH_CODE = forms.CharField(label="BRANCH_CODE", max_length=50)


def index(request):
    return render(request, 'prediction/index.html')


def signin(request):
    return render(request, 'prediction/signin.html')


@csrf_exempt
def authentication(request):
    if request.method == 'POST':
        uf = UserForm(request.POST)
        if uf.is_valid():
            email = uf.cleaned_data['email']
            password = uf.cleaned_data['password']
            user = User.objects.filter(email=email, password=password).get()
            if user:
                response = HttpResponseRedirect('predictor.html')
                response.set_cookie('email', email, 3600)
                response.set_cookie('username', user.username)
                return response
            else:
                return render(request, 'prediction/error.html')
    else:
        uf = UserForm()
    return render(request, 'prediction/error.html')


def signup(request):
    return render(request, 'prediction/signup.html')


@csrf_exempt
def addUser(request):
    Method = request.method
    if Method == 'POST':
        uf = newUserForm(request.POST)
        if uf.is_valid():
            username = uf.cleaned_data['username']
            password = uf.cleaned_data['password']
            email = uf.cleaned_data['email']
            try:
                registJudge = User.objects.filter(email=email).get()
                return render(request, 'prediction/signup.html', {'msg': "The account already existed!"})
            except:
                registAdd = User.objects.create(username=username, password=password,email=email)
                context = {'registAdd': registAdd}
                return render(request, 'prediction/signin.html', context)
        else:
            return render(request, 'prediction/error.html')
    else:
        return render(request, 'prediction/signup.html')


def signout(request):
    return render(request, 'prediction/index.html')


@csrf_exempt
def predictor(request):
    email = request.COOKIES.get('email', '')
    context = {'email': email}
    return render(request, 'prediction/predictor.html', context)

@csrf_exempt
def predictor_branch(request):
    email = request.COOKIES.get('email', '')
    context = {'email': email}
    return render(request, 'prediction/predictor_branch.html', context)

@csrf_exempt
def profile(request):
    email = request.COOKIES.get('email', '')
    username = request.COOKIES.get('username', '')
    context = {'email': email, 'username': username}
    return render(request, 'prediction/profile.html', context)


@csrf_exempt
def success(request):
    email = request.COOKIES.get('email', '')
    context = {'email': email}
    return render(request, 'prediction/success.html', context)


@csrf_exempt
def result(request):
    if request.method == 'POST':
        form = itemInfoForm(request.POST)
        if form.is_valid():
            SRV_HUB = form.cleaned_data['SRV_HUB']
            BRANCH_CODE = form.cleaned_data['BRANCH_CODE']
            INV_ITEM_ID = form.cleaned_data['INV_ITEM_ID']
            MODEL = form.cleaned_data['MODEL']
            email = request.COOKIES.get('email', '')
            try:
                item = Hub17.objects.filter(BRANCH_CODE=BRANCH_CODE, INV_ITEM_ID=INV_ITEM_ID, SRV_HUB=SRV_HUB).get()
                if MODEL == "LINEAR_REGRESSION":
                    modelContext = load_linear(item)
                    modelName = "LINEAR REGRESSION"
                elif MODEL == "TIME_SERIES_ARIMA":
                    # modelContext = time_series(item)
                    modelContext = load_time(item)
                    modelName = "TIME SERIES(ARIMA)"
                else:
                    modelContext = load_time(item)
                    modelName = "TIME SERIES(LSTM)"
                context = {'item': item, 'email': email, 'modelName': modelName, 'modelContext': modelContext}
                return render(request, 'prediction/result.html', context)
            except:
                return render(request, 'prediction/predictor.html', {'msg': 'Invalid item1!', 'email': email})
        else:
            return render(request, 'prediction/predictor.html', {'msg': 'Invalid item!', 'email': request.COOKIES.get('email', '')})
    else:
        return render(request, 'prediction/predictor.html')


@csrf_exempt
def branch_result(request):
    if request.method == 'GET':
        form = branchInfoForm(request.GET)
        if form.is_valid():
            SRV_HUB = form.cleaned_data['SRV_HUB']
            BRANCH_CODE = form.cleaned_data['BRANCH_CODE']
            email = request.COOKIES.get('email', '')
            try:
                items = Hub17_Result.objects.filter(BRANCH_CODE=BRANCH_CODE, SRV_HUB=SRV_HUB)
                paginator = Paginator(items, 100)
                page = request.GET.get('page')
                try:
                    sayfalar = paginator.page(page)
                except PageNotAnInteger:
                    sayfalar = paginator.page(1)
                except EmptyPage:
                    sayfalar = paginator.page(paginator.num_pages)
                context = {'email': email, 'branch': BRANCH_CODE, 'hub': SRV_HUB, 'itemsOnePage': sayfalar}
                #save_results(sayfalar)
                # context = {'items': items, 'email': email, 'branch': BRANCH_CODE}
                return render(request, 'prediction/branch_result.html', context)
            except:
                return render(request, 'prediction/predictor.html', {'msg': 'Invalid item1!', 'email': email})
        else:
            return render(request, 'prediction/predictor.html', {'msg': 'Invalid item!', 'email': request.COOKIES.get('email', '')})
    else:
        return render(request, 'prediction/predictor_branch.html')


@csrf_exempt
def report(request):
    if request.method == 'POST':
        form = branchInfoForm(request.POST)
        if form.is_valid():
            SRV_HUB = form.cleaned_data['SRV_HUB']
            BRANCH_CODE = form.cleaned_data['BRANCH_CODE']
            try:
                #items = Hub17.objects.filter(BRANCH_CODE=BRANCH_CODE, SRV_HUB=SRV_HUB)
                items = Hub17_Result.objects.filter(BRANCH_CODE=BRANCH_CODE, SRV_HUB=SRV_HUB)
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="report.csv"'
                writer = csv.writer(response)
                writer.writerow(['id', 'INV_ITEM_ID', 'BRANCH_CODE', 'SRV_HUB', 'PREDICTION1',
                                    'PREDICTION2', 'PREDICTION3', 'PREDICTION4', 'ALGORITHM'])
                for item in items:
                    origin = Hub17.objects.filter(BRANCH_CODE=BRANCH_CODE, SRV_HUB=SRV_HUB, INV_ITEM_ID=item.INV_ITEM_ID)
                    origin = origin[0]
                    months = [origin.OCT2012, origin.NOV2012, origin.DEC2012, origin.JAN2013, origin.FEB2013, origin.MAR2013,
                                        origin.APR2013,
                                        origin.MAY2013, origin.JUN2013, origin.JUL2013, origin.AUG2013, origin.SEP2013, origin.OCT2013,
                                        origin.NOV2013,
                                        origin.DEC2013, origin.JAN2014, origin.FEB2014, origin.MAR2014, origin.APR2014, origin.MAY2014,
                                        origin.JUN2014,
                                        origin.JUL2014, origin.AUG2014, origin.SEP2014, origin.OCT2014, origin.NOV2014, origin.DEC2014,
                                        origin.JAN2015,
                                        origin.FEB2015, origin.MAR2015, origin.APR2015, origin.MAY2015, origin.JUN2015, origin.JUL2015,
                                        origin.AUG2015,
                                        origin.SEP2015, origin.OCT2015, origin.NOV2015, origin.DEC2015, origin.JAN2016, origin.FEB2016,
                                        origin.MAR2016,
                                        origin.APR2016, origin.MAY2016, origin.JUN2016, origin.JUL2016, origin.AUG2016,
                                        origin.SEP2016]
                    count = 0
                    sums = 0
                    for month in months:
                        sums = sums + month
                        if month <= 0:
                            count = count + 1
                    mean = round(sums / 48)
                    if count >= 43:
                        writer.writerow([item.id, item.INV_ITEM_ID, item.BRANCH_CODE, item.SRV_HUB, mean,
                                         mean, mean, mean, "MEAN1"])
                    elif item.LR_R2 >= 0.1:
                        writer.writerow([item.id, item.INV_ITEM_ID, item.BRANCH_CODE, item.SRV_HUB, item.LR_JUN2016,
                                         item.LR_JUL2016, item.LR_AUG2016, item.LR_SEP2016, "Linear Regression"])
                    elif item.AR_R2 >= 0.1:
                        writer.writerow([item.id, item.INV_ITEM_ID, item.BRANCH_CODE, item.SRV_HUB, item.AR_JUN2016,
                                         item.AR_JUL2016, item.AR_AUG2016, item.AR_SEP2016, "ARIMA"])
                    else:
                        writer.writerow([item.id, item.INV_ITEM_ID, item.BRANCH_CODE, item.SRV_HUB, mean,
                                             mean, mean, mean, "MEAN2"])
                return response
            except:
                return render(request, 'prediction/error.html')
        else:
            return render(request, 'prediction/error.html')
    else:
        return render(request, 'prediction/error.html')


def linear_regression(item):
    X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12],
         [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
         [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36],
         [37], [38], [39], [40], [41], [42], [43]]
    y = [item.OCT2012, item.NOV2012, item.DEC2012, item.JAN2013, item.FEB2013, item.MAR2013, item.APR2013, item.MAY2013,
         item.JUN2013, item.JUL2013, item.AUG2013, item.SEP2013, item.OCT2013, item.NOV2013, item.DEC2013, item.JAN2014,
         item.FEB2014, item.MAR2014, item.APR2014, item.MAY2014, item.JUN2014, item.JUL2014, item.AUG2014, item.SEP2014,
         item.OCT2014, item.NOV2014, item.DEC2014, item.JAN2015, item.FEB2015, item.MAR2015, item.APR2015, item.MAY2015,
         item.JUN2015, item.JUL2015, item.AUG2015, item.SEP2015, item.OCT2015, item.NOV2015, item.DEC2015, item.JAN2016,
         item.FEB2016, item.MAR2016, item.APR2016, item.MAY2016]
    lr = linear_model.LinearRegression()
    start = time.clock()
    lr.fit(X, y)
    predictionX = [[44], [45], [46], [47]]
    actualY = [round(item.JUN2016), round(item.JUL2016), round(item.AUG2016), round(item.SEP2016)]
    predictionY = lr.predict(predictionX)
    for i in range(len(predictionY)):
        predictionY[i] = round(predictionY[i])
    coefficient_of_dermination = round(lr.score(X, y), 3)
    mse = mean_squared_error(actualY, predictionY)
    rmse = round(sqrt(mse), 3)
    # errors = 0
    # for x in range(len(predictionY)):
    #     a1 = predictionY[x] - actualY[x]
    #     a2 = actualY[x] * actualY[x]
    #     errors = errors + a1 * a1 / (a2 * a2)
    #     # errors = errors + (predictionY[x] - actualY[x]) * (predictionY[x] - actualY[x]) / (actualY[x] * actualY[x])
    # errors = round(errors / len(predictionY), 3)
    end = time.clock()

    plt.scatter(X, y, color='blue')
    plt.scatter(predictionX, predictionY, color='red')
    plt.scatter(predictionX, actualY, color='blue')
    #plt.plot(X, y, color='blue')
    # plt.plot(predictionX, predictionY, color='yellow')
    # plt.plot(predictionX, actualY, color='blue')
    plt.plot(X, lr.predict(X), color='red', linewidth=4)

    path = "/Users/Iris/PycharmProjects/inventoryPrediction/inventoryPrediction/static/resultImgs/"
    plt.savefig(path + item.BRANCH_CODE+"_"+item.INV_ITEM_ID+"_lr"+".png")
    imagePath = "/static/resultImgs/"+item.BRANCH_CODE+"_"+item.INV_ITEM_ID+"_lr"+".png"
    plt.close()

    # item_result = Hub17_Result.objects.filter().get(id=item.id)
    # item_result.LR_JUN2016 = predictionY[0]
    # item_result.LR_JUL2016 = predictionY[1]
    # item_result.LR_AUG2016 = predictionY[2]
    # item_result.LR_SEP2016 = predictionY[3]
    # item_result.LR_R2 = coefficient_of_dermination
    # item_result.LR_RMSE = rmse
    # item_result.LR_TIME = round(end - start, 3)
    # item_result.save()

    linearContext = {'predictionY0': predictionY[0], 'predictionY1': predictionY[1],
                     'predictionY2': predictionY[2], 'predictionY3': predictionY[3],
                     'actualY0': actualY[0], 'actualY1': actualY[1], 'actualY2': actualY[2], 'actualY3': actualY[3],
                     'time': round(end - start, 3), 'coefficient_of_dermination': coefficient_of_dermination,
                     'rmse': rmse, 'imagePath': imagePath}
    return linearContext


def time_series(item):
    series = Series([item.OCT2012, item.NOV2012, item.DEC2012, item.JAN2013, item.FEB2013, item.MAR2013, item.APR2013,
                     item.MAY2013, item.JUN2013, item.JUL2013, item.AUG2013, item.SEP2013, item.OCT2013, item.NOV2013,
                     item.DEC2013, item.JAN2014, item.FEB2014, item.MAR2014, item.APR2014, item.MAY2014, item.JUN2014,
                     item.JUL2014, item.AUG2014, item.SEP2014, item.OCT2014, item.NOV2014, item.DEC2014, item.JAN2015,
                     item.FEB2015, item.MAR2015, item.APR2015, item.MAY2015, item.JUN2015, item.JUL2015, item.AUG2015,
                     item.SEP2015, item.OCT2015, item.NOV2015, item.DEC2015, item.JAN2016, item.FEB2016, item.MAR2016,
                     item.APR2016, item.MAY2016, item.JUN2016, item.JUL2016, item.AUG2016, item.SEP2016])
    dataset = series[0:44]
    validation = series[44:48]

    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

        # evaluate an ARIMA model for a given order (p,d,q) and return RMSE
    def evaluate_arima_model(X, arima_order):
        # prepare training dataset
        X = X.astype('float32')
        train_size = int(len(X) * 0.50)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            # difference data
            months_in_year = 12
            diff = difference(history, months_in_year)
            try:
                model = ARIMA(diff, order=arima_order)
            except Exception as e:
                print(str(e))
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            yhat = inverse_difference(history, yhat, months_in_year)
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mse)
        return rmse

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        mse = evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                    except:
                        continue
        return best_cfg

    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    best_cfg = evaluate_models(dataset.values, p_values, d_values, q_values)

    def getBias(best_cfg):
        # prepare data
        X = dataset.values
        X = X.astype('float32')
        train_size = int(len(X) * 0.50)
        train, test = X[0:train_size], X[train_size:]
        # walk-forward validation
        history = [x for x in train]
        predictions = list()
        for i in range(len(test)):
            # difference data
            months_in_year = 12
            diff = difference(history, months_in_year)
            # predict
            model = ARIMA(diff, order=best_cfg)
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            yhat = inverse_difference(history, yhat, months_in_year)
            predictions.append(yhat)
            # observation
            obs = test[i]
            history.append(obs)
        # errors
        residuals = [test[i] - predictions[i] for i in range(len(test))]
        residuals = DataFrame(residuals)
        return residuals.mean()

    # load and prepare datasets
    X = dataset.values.astype('float32')
    history = [x for x in X]
    months_in_year = 12
    y = validation.values.astype('float32')
    try:
        bias = getBias(best_cfg)
    except:
        timeContext = {'predictionY0': round(item.JUN2016),
                       'predictionY1': round(item.JUL2016),
                       'predictionY2': round(item.AUG2016),
                       'predictionY3': round(item.SEP2016),
                       'actualY0': round(item.JUN2016), 'actualY1': round(item.JUL2016), 'actualY2': round(item.AUG2016),
                       'actualY3': round(item.SEP2016),
                       'time': round(-78, 3),
                       'coefficient_of_dermination': round(0, 3),
                       'rmse': round(0, 3), 'imagePath': 'imagePath'
                       }
        # item_result = Hub17_Result.objects.filter().get(id=item.id)
        # item_result.AR_JUN2016 = round(item.JUN2016)
        # item_result.AR_JUL2016 = round(item.JUL2016)
        # item_result.AR_AUG2016 = round(item.AUG2016)
        # item_result.AR_SEP2016 = round(item.SEP2016)
        # item_result.AR_R2 = round(0, 3)
        # item_result.AR_RMSE = round(0, 3)
        # item_result.AR_TIME = round(-78, 3)
        # item_result.save()
        return timeContext

    try:
        start = time.clock()
        # make first prediction
        predictions = list()
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=best_cfg)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = float(model_fit.forecast()[0])
        yhat = bias + inverse_difference(history, yhat, months_in_year)
        predictions.append(yhat)
        history.append(y[0])
        # rolling forecasts
        for i in range(1, len(y)):
            # difference data
            months_in_year = 12
            diff = difference(history, months_in_year)
            # predict
            model = ARIMA(diff, order=best_cfg)
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            yhat = bias + inverse_difference(history, yhat, months_in_year)
            predictions.append(yhat)
            # observation
            obs = y[i]
            history.append(obs)
        # report performance
        roundPredictions = [round(predictions[0].get_value(0)), round(predictions[1].get_value(0)),
                            round(predictions[2].get_value(0)), round(predictions[3].get_value(0))]
        mse = mean_squared_error(y, roundPredictions)
        coefficient_of_dermination = round(r2_score(y, roundPredictions), 3)
        rmse = round(sqrt(mse), 3)
        # errors = 0
        # for i in range(len(roundPredictions)):
        #     a1 = (roundPredictions[i] - y[i]) * (roundPredictions[i] - y[i])
        #     a2 = (y[i] * y[i])
        #     # errors = errors + (roundPredictions[i] - y[i]) * (roundPredictions[i] - y[i]) / (y[i] * y[i])
        #     errors = errors + a1 * a1 / (a2 * a2)
        # errors = round(errors / len(roundPredictions), 3)
        end = time.clock()

    except:
        timeContext = {'predictionY0': round(item.JUN2016),
                       'predictionY1': round(item.JUL2016),
                       'predictionY2': round(item.AUG2016),
                       'predictionY3': round(item.SEP2016),
                       'actualY0': round(item.JUN2016), 'actualY1': round(item.JUL2016),
                       'actualY2': round(item.AUG2016),
                       'actualY3': round(item.SEP2016),
                       'time': round(-78, 3),
                       'coefficient_of_dermination': round(0, 3),
                       'rmse': round(0, 3), 'imagePath': 'imagePath'
                       }
        # item_result = Hub17_Result.objects.filter().get(id=item.id)
        # item_result.AR_JUN2016 = round(item.JUN2016)
        # item_result.AR_JUL2016 = round(item.JUL2016)
        # item_result.AR_AUG2016 = round(item.AUG2016)
        # item_result.AR_SEP2016 = round(item.SEP2016)
        # item_result.AR_R2 = round(0, 3)
        # item_result.AR_RMSE = round(0, 3)
        # item_result.AR_TIME = round(-78, 3)
        # item_result.save()
        return timeContext


    # output
    plt.plot(series, color='blue')
    plt.plot([44, 45, 46, 47], predictions, color='red')
    path = "/Users/Iris/PycharmProjects/inventoryPrediction/inventoryPrediction/static/resultImgs/"
    plt.savefig(path + item.BRANCH_CODE + "_" + item.INV_ITEM_ID + "_ar" + ".png")
    imagePath = "/static/resultImgs/" + item.BRANCH_CODE + "_" + item.INV_ITEM_ID + "_ar" + ".png"
    plt.close()

    # item_result = Hub17_Result.objects.filter().get(id=item.id)
    # item_result.AR_JUN2016 = roundPredictions[0]
    # item_result.AR_JUL2016 = roundPredictions[1]
    # item_result.AR_AUG2016 = roundPredictions[2]
    # item_result.AR_SEP2016 = roundPredictions[3]
    # item_result.AR_R2 = coefficient_of_dermination
    # item_result.AR_RMSE = rmse
    # item_result.AR_TIME = round(end - start, 3)
    # item_result.save()

    timeContext = {'predictionY0': roundPredictions[0], 'predictionY1': roundPredictions[1],
            'predictionY2': roundPredictions[2], 'predictionY3': roundPredictions[3],
            'actualY0': item.JUN2016, 'actualY1': item.JUL2016, 'actualY2': item.AUG2016, 'actualY3': item.SEP2016,
            'time': round(end-start, 3), 'coefficient_of_dermination': coefficient_of_dermination,
            'rmse': rmse, 'imagePath': imagePath
    }
    return timeContext


def load_linear(item):
    result = Hub17_Result.objects.filter(BRANCH_CODE=item.BRANCH_CODE, SRV_HUB=item.SRV_HUB, INV_ITEM_ID=item.INV_ITEM_ID)
    result = result[0]
    linearContext = {'predictionY0': result.LR_JUN2016,
                   'predictionY1': result.LR_JUL2016,
                   'predictionY2': result.LR_AUG2016,
                   'predictionY3': result.LR_SEP2016,
                   'actualY0': result.JUN2016,
                   'actualY1': result.JUL2016,
                   'actualY2': result.AUG2016,
                   'actualY3': result.SEP2016,
                   'time': result.LR_TIME,
                   'coefficient_of_dermination': result.LR_R2,
                   'rmse': result.LR_RMSE,
                   'imagePath': 'imagePath',
                   'item': item
                   }
    return linearContext


def load_time(item):
    result = Hub17_Result.objects.filter(BRANCH_CODE=item.BRANCH_CODE, SRV_HUB=item.SRV_HUB, INV_ITEM_ID=item.INV_ITEM_ID)
    result = result[0]
    timeContext = {'predictionY0': result.AR_JUN2016,
                   'predictionY1': result.AR_JUL2016,
                   'predictionY2': result.AR_AUG2016,
                   'predictionY3': result.AR_SEP2016,
                   'actualY0': result.JUN2016,
                   'actualY1': result.JUL2016,
                   'actualY2': result.AUG2016,
                   'actualY3': result.SEP2016,
                   'time': result.AR_TIME,
                   'coefficient_of_dermination': result.AR_R2,
                   'rmse': result.AR_RMSE,
                   'imagePath': 'imagePath',
                   'item': item
                   }
    return timeContext
# def lstm(item):
#     series = Series([item.OCT2012, item.NOV2012, item.DEC2012, item.JAN2013, item.FEB2013, item.MAR2013, item.APR2013,
#                      item.MAY2013, item.JUN2013, item.JUL2013, item.AUG2013, item.SEP2013, item.OCT2013, item.NOV2013,
#                      item.DEC2013, item.JAN2014, item.FEB2014, item.MAR2014, item.APR2014, item.MAY2014, item.JUN2014,
#                      item.JUL2014, item.AUG2014, item.SEP2014, item.OCT2014, item.NOV2014, item.DEC2014, item.JAN2015,
#                      item.FEB2015, item.MAR2015, item.APR2015, item.MAY2015, item.JUN2015, item.JUL2015, item.AUG2015,
#                      item.SEP2015, item.OCT2015, item.NOV2015, item.DEC2015, item.JAN2016, item.FEB2016, item.MAR2016,
#                      item.APR2016, item.MAY2016, item.JUN2016, item.JUL2016, item.AUG2016, item.SEP2016])
#     dataset = series[0:44]
#     validation = series[44:48]
#
#     # frame a sequence as a supervised learning problem
#     def timeseries_to_supervised(data, lag=1):
#         df = DataFrame(data)
#         columns = [df.shift(i) for i in range(1, lag + 1)]
#         columns.append(df)
#         df = concat(columns, axis=1)
#         df.fillna(0, inplace=True)
#         return df
#
#     # create a differenced series
#     def difference(dataset, interval=1):
#         diff = list()
#         for i in range(interval, len(dataset)):
#             value = dataset[i] - dataset[i - interval]
#             diff.append(value)
#         return Series(diff)
#
#     # invert differenced value
#     def inverse_difference(history, yhat, interval=1):
#         return yhat + history[-interval]
#
#     # scale train and test data to [-1, 1]
#     def scale(train, test):
#         # fit scaler
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         scaler = scaler.fit(train)
#         # transform train
#         train = train.reshape(train.shape[0], train.shape[1])
#         train_scaled = scaler.transform(train)
#         # transform test
#         test = test.reshape(test.shape[0], test.shape[1])
#         test_scaled = scaler.transform(test)
#         return scaler, train_scaled, test_scaled
#
#     # inverse scaling for a forecasted value
#     def invert_scale(scaler, X, value):
#         new_row = [x for x in X] + [value]
#         array = numpy.array(new_row)
#         array = array.reshape(1, len(array))
#         inverted = scaler.inverse_transform(array)
#         return inverted[0, -1]
#
#     # fit an LSTM network to training data
#     def fit_lstm(train, batch_size, nb_epoch, neurons):
#         X, y = train[:, 0:-1], train[:, -1]
#         X = X.reshape(X.shape[0], 1, X.shape[1])
#         model = Sequential()
#         model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
#         model.add(Dense(1))
#         model.compile(loss='mean_squared_error', optimizer='adam')
#         for i in range(nb_epoch):
#             model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
#             model.reset_states()
#         return model
#
#     # make a one-step forecast
#     def forecast_lstm(model, batch_size, X):
#         X = X.reshape(1, 1, len(X))
#         yhat = model.predict(X, batch_size=batch_size)
#         return yhat[0, 0]
#
#     start = time.clock()
#     # transform data to be stationary
#     raw_values = series.values
#     #diff_values = raw_values.diff()
#     diff_values = difference(raw_values, 1)
#
#     # transform data to be supervised learning
#     supervised = timeseries_to_supervised(diff_values, 1)
#     supervised_values = supervised.values
#
#     # split data into train and test-sets
#     train, test = supervised_values[0:-4], supervised_values[-4:]
#
#     # transform the scale of the data
#     scaler, train_scaled, test_scaled = scale(train, test)
#
#     # fit the model
#     lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
#     # forecast the entire training dataset to build up state for forecasting
#     train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#     lstm_model.predict(train_reshaped, batch_size=1)
#
#     # walk-forward validation on the test data
#     predictions = list()
#     for i in range(len(test_scaled)):
#         # make one-step forecast
#         X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#         yhat = forecast_lstm(lstm_model, 1, X)
#         # invert scaling
#         yhat = invert_scale(scaler, X, yhat)
#         # invert differencing
#         yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
#         # store forecast
#         predictions.append(yhat)
#         expected = raw_values[len(train) + i + 1]
#
#     # report performance
#     rmse = sqrt(mean_squared_error(raw_values[-4:], predictions))
#     coefficient_of_dermination = r2_score(raw_values[-4:], predictions)
#     end = time.clock()
#
#     #output
#     plt.plot(series, color='blue')
#     plt.plot([44, 45, 46, 47], predictions, color='red')
#     path = "/Users/Iris/PycharmProjects/inventoryPrediction/inventoryPrediction/static/resultImgs/"
#     plt.savefig(path + item.BRANCH_CODE + "_" + item.INV_ITEM_ID + "_ls" + ".png")
#     imagePath = "/static/resultImgs/" + item.BRANCH_CODE + "_" + item.INV_ITEM_ID + "_ls" + ".png"
#     plt.close()
#     timeContext = {'predictionY0': round(predictions[0].get_value(0), 3), 'predictionY1': round(predictions[1].get_value(0), 3),
#             'predictionY2': round(predictions[2].get_value(0), 3), 'predictionY3': round(predictions[3].get_value(0), 3),
#             'actualY0': item.JUN2016, 'actualY1': item.JUL2016, 'actualY2': item.AUG2016, 'actualY3': item.SEP2016,
#             'time': round(end-start, 3), 'coefficient_of_dermination': round(coefficient_of_dermination, 3),
#             'rmse': round(rmse, 3), 'imagePath': imagePath
#     }
#     return timeContext


# def save_results(items):
#     for item in items:
#         if item.id > 526:
#             time_series(item)



