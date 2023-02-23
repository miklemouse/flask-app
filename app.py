from flask import Flask, render_template, send_file, request
import matplotlib.pyplot as plt

import pandas as pd

import time

import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objects as go

import os

import pickle

#preprocess

def my_setup():
    if os.path.exists('data/tmp/cached'):
        with open('data/tmp/cached', 'rb') as file:
            data = pickle.load(file)
        return data


    if not os.path.exists('data/country_vaccinations_pr.csv'):
        vax = pd.read_csv('data/country_vaccinations.csv')


        vax.drop(['vaccines', 'source_name', 'source_website',
                  'daily_vaccinations', 'daily_vaccinations_raw',
                  'daily_vaccinations_per_million'],
                  axis=1, inplace=True)

        assert vax.equals(vax.sort_values(['country', 'date']))

        dates = sorted(set(vax['date']))
        countries = list(OrderedDict.fromkeys((vax['country'])))
        iso_codes = list(OrderedDict.fromkeys((vax['iso_code'])))

        dates_ = dates * len(countries)
        countries_ = []
        for country in countries:
            countries_ += [country] * len(dates)

        iso_codes_ = []
        for code in iso_codes:
            iso_codes_ += [code] * len(dates)


        nans = [float('NaN')] * len(dates_)


        vax_= pd.DataFrame({'country':countries_, 'iso_code':iso_codes_, 'date':dates_,
                            'total_vaccinations':nans,
                            'people_vaccinated':nans,
                            'people_fully_vaccinated':nans,
                            'total_vaccinations_per_hundred':nans,
                            'people_vaccinated_per_hundred':nans,
                            'people_fully_vaccinated_per_hundred':nans})

        for index, row in vax_.iterrows():
            if index % len(dates) == 0:
                by_country = vax.loc[vax['country'] == row['country']]
            if not by_country.loc[by_country['date'] == row['date']].empty:
                x = by_country.loc[by_country['date'] == row['date']].iloc[0]
                vax_.iloc[index] = x
            if index % 10000 == 0:
                print(f'Takes around 5 min. Progress: {int(index / len(dates_) * 100)}%')
        print('Just a little more...')

        vax = vax_

        for param in set(vax.columns) - {'country', 'iso_code', 'date'}:
            for index, row in vax.iterrows():
                if pd.isnull(row[param]):
                    if index > 0 and vax.at[index - 1, 'country'] == row['country']:
                        vax.at[index, param] = vax.at[index - 1, param]
                    else:
                        vax.at[index, param] = 0

        vax.to_csv('data/country_vaccinations_pr.csv', index=False)

    else:
        vax = pd.read_csv('data/country_vaccinations_pr.csv')

        dates = sorted(set(vax['date']))
        countries = list(OrderedDict.fromkeys((vax['country'])))
        iso_codes = list(OrderedDict.fromkeys((vax['iso_code'])))


    country_info = pd.read_csv('data/country-codes.csv')

    region_by_iso = dict()
    devtype_by_iso = dict()

    for index, row in country_info.iterrows():
        region_by_iso[row['ISO3166-1-Alpha-3']] = row['Region Name']
        devtype_by_iso[row['ISO3166-1-Alpha-3']] = row['Developed / Developing Countries']

    up_to_date_vax = pd.DataFrame(columns=vax.columns)

    for country in sorted(set(vax['country'])):
        up_to_date_vax = up_to_date_vax.append(vax.loc[vax['country'] == country].iloc[-1])

    population_by_iso = dict()
    for index, row in up_to_date_vax.iterrows():
        population_by_iso[row['iso_code']] = int(row['people_vaccinated'] / row['people_vaccinated_per_hundred'] * 100)

    cols = ['total_vaccinations_per_hundred',
            'people_vaccinated_per_hundred', 
            'people_fully_vaccinated_per_hundred']

    stats = pd.DataFrame(columns=cols)

    # mean
    stats = stats.append(up_to_date_vax[cols].mean(), sort=True, ignore_index=True)

    # median
    stats = stats.append(up_to_date_vax[cols].median(), sort=True, ignore_index=True)

    # standard deviation
    stats = stats.append(up_to_date_vax[cols].std(), sort=True, ignore_index=True)

    # rename the columns
    stats = stats.rename(index={0: 'mean', 1: 'median', 2: 'std'})


    population = 0
    for _, pop in population_by_iso.items():
        population += pop

    perc_by_date = OrderedDict([(date, 0) for date in dates])

    for index, row in vax.iterrows():
        perc_by_date[row['date']] += row['people_fully_vaccinated']

    for date in dates:
        perc_by_date[date] /= (population / 100)

    regions = set(region_by_iso.values())
    population_by_region = dict([(region, 0) for region in regions])

    for iso, pop in population_by_iso.items():
        if iso in region_by_iso:
            population_by_region[region_by_iso[iso]] += pop

    perc_by_reg = dict([(region, OrderedDict([(date, 0) for date in dates])) for region in regions])

    for index, row in vax.iterrows():
        if row['iso_code'] not in region_by_iso:
            continue
        reg = region_by_iso[row['iso_code']]
        perc_by_reg[reg][row['date']] += row['people_fully_vaccinated']

    for region in regions:
        for date in dates:
            perc_by_reg[region][date] /= (population_by_region[region] / 100)

    devtypes = set(devtype_by_iso.values())
    population_by_dev = dict([(dev, 0) for dev in devtypes])

    for iso, pop in population_by_iso.items():
        if iso in devtype_by_iso:
            population_by_dev[devtype_by_iso[iso]] += pop


    perc_by_dev = dict([(devtype, OrderedDict([(date, 0) for date in dates])) for devtype in devtypes])

    for index, row in vax.iterrows():
        if row['iso_code'] not in devtype_by_iso:
            continue
        dev = devtype_by_iso[row['iso_code']]
        perc_by_dev[dev][row['date']] += row['people_fully_vaccinated']

    for dev in devtypes:
        for date in dates:
            perc_by_dev[dev][date] /= (population_by_dev[dev] / 100)

    cols = ['total_vaccinations',
            'people_vaccinated', 
            'people_fully_vaccinated']

    total = pd.DataFrame(index = list(range(len(set(vax['date'])))), columns=['date']+cols)

    i = 0
    for date in sorted(set(vax['date'])):
        total.iloc[i]['date'] = date
        total.iloc[i][cols] = vax.loc[vax['date'] == date].sum(numeric_only=True)[cols]
        i += 1



    data = (stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total,
            population_by_region, population_by_dev)

    with open('data/tmp/cached', 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    return data


app = Flask(__name__)



links = {"Download" : "/download",
         "Download analysis in PDF" : "/analysis",
         "View Raw Data" : "/view_data",
         "Descriptive stats" : "/stats",
         "1. Fully vaxed perc worldwide [LINE]" : "/graph1",
         "2. Fully vaxed perc by region [LINE]" : "/graph2",
         "3. Fully vaxed perc by dev type [LINE]" : "/graph3",
         "4. Up to date vaxed perc by region [BAR]" : "/graph4",
         "5. Up to date vaxed perc by dev type [BAR]" : "/graph5",
         "6. Vax stats in total worldwide [LINE]" : "/graph6",
         "7. Up to date vaxed people in total by region [PIE]" : "/graph7",
         "8. Up to date vaxed people in total by region [PIE]" : "/graph8",
         "Technical: Download cache" : "/cache"}


def render_index (image=None, html_string=None, filters=None,  errors=None, current_filter_value=""):
    return render_template("index.html", links=links, image=image, code=time.time(), html_string=html_string,
                           filters=filters, errors=errors, current_filter_value=current_filter_value)

@app.route('/', methods=['GET'])
def main_page():
    return render_index()

@app.route(links["Download"], methods=['GET'])
def download_data():
    return send_file("data/country_vaccinations.csv", as_attachment=True)

@app.route(links["Download analysis in PDF"], methods=['GET'])
def download_analysis():
    return send_file("data/Vaccination_Analysis.pdf", as_attachment=True)

@app.route(links["View Raw Data"], methods=['GET'])
def view_data():
    df = pd.read_csv("data/country_vaccinations.csv")
    errors = []
    current_filter_value = ""
    if request.method == "POST":
        current_filter = request.form.get('filters')
        current_filter_value = current_filter
        if current_filter:
            try:
                df = df.query(current_filter)
            except Exception as e:
                errors.append('<font color="red">Incorrect filter</font>')
                print(e)

    html_string = df.to_html()
    return render_index(html_string=html_string, filters=True, errors=errors, current_filter_value=current_filter_value)

@app.route(links["Descriptive stats"], methods=['GET'])
def desc_stats():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()
    html_string = stats.to_html()
    return render_index(html_string=html_string)


@app.route(links["1. Fully vaxed perc worldwide [LINE]"], methods=['GET'])
def graph1():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    line = go.Scatter(name='vax, %',
                      x=dates,
                      y=list(perc_by_date.values()))

    fig = go.Figure(line)


    fig.update_layout(title='Percentage of fully vaccinated people worldwide',
                      yaxis_title='Vaxed people, %')

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["2. Fully vaxed perc by region [LINE]"], methods=['GET'])
def graph2():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    lines = [go.Scatter(name=region,
                        x=dates,
                        y=list(perc_by_reg[region].values()))
             for region in regions if type(region) is str]

    fig = go.Figure(lines)

    fig.update_layout(title='Percentage of fully vaccinated people in different regions',
                      yaxis_title='Vaxed people, %')

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["3. Fully vaxed perc by dev type [LINE]"], methods=['GET'])
def graph3():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    lines = [go.Scatter(name=devtype,
                        x=dates,
                        y=list(perc_by_dev[devtype].values()))
             for devtype in devtypes if type(devtype) is str]

    fig = go.Figure(lines)

    fig.update_layout(title='Percentage of fully vaccinated people by country development type',
                      yaxis_title='Vaxed people, %')

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))


@app.route(links["4. Up to date vaxed perc by region [BAR]"], methods=['GET'])
def graph4():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    sorted_reg = sorted([reg for reg in regions if type(reg) is str])

    bar = go.Bar(x=sorted_reg,
                 y=[list(perc_by_reg[reg].values())[-1]
                    for reg in sorted_reg])

    fig = go.Figure(bar)

    fig.update_layout(title='Up to date percentage of fully vaccinated people in different regions',
                      yaxis_title='Vaxed people, %')

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["5. Up to date vaxed perc by dev type [BAR]"], methods=['GET'])
def graph5():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    sorted_dev = sorted([dev for dev in devtypes if type(dev) is str])

    bar = go.Bar(x=sorted_dev,
                 y=[list(perc_by_dev[dev].values())[-1]
                    for dev in sorted_dev])

    fig = go.Figure(bar)

    fig.update_layout(title='Up to date percentage of fully vaccinated people by country dev type',
                      yaxis_title='Vaxed people, %')


    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["6. Vax stats in total worldwide [LINE]"], methods=['GET'])
def graph6():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    lines = [go.Scatter(name=col.replace('_',' '),
                        x=dates,
                        y=total[col])
             for col in cols]

    fig = go.Figure(lines)

    fig.update_layout(title='Global statistics',
                      yaxis_title='Number of people / vaccinations',)

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["7. Up to date vaxed people in total by region [PIE]"], methods=['GET'])
def graph7():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    sorted_reg = sorted([reg for reg in regions if type(reg) is str])

    pie = go.Pie(labels=sorted_reg,
                 values=[list(perc_by_reg[reg].values())[-1] / 100 * population_by_region[reg]
                         for reg in sorted_reg])

    fig = go.Figure(pie)

    fig.update_layout(title='Fully vaccinated people in total by region')

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["8. Up to date vaxed people in total by region [PIE]"], methods=['GET'])
def graph8():
    stats, dates, perc_by_date, regions, perc_by_reg, devtypes, perc_by_dev, cols, total, population_by_region, population_by_dev = my_setup()

    sorted_dev = sorted([dev for dev in devtypes if type(dev) is str])

    pie = go.Pie(labels=sorted_dev,
                 values=[list(perc_by_dev[dev].values())[-1] / 100 * population_by_dev[dev]
                    for dev in sorted_dev])

    fig = go.Figure(pie)

    fig.update_layout(title="""Fully vaccinated people in total by country dev type.""")

    return render_index(html_string = fig.to_html(full_html=False, include_plotlyjs='cdn'))

@app.route(links["Technical: Download cache"], methods=['GET'])
def download_cache():
    return send_file("data/tmp/cached", as_attachment=True)



if __name__ == '__main__':
    app.run()
