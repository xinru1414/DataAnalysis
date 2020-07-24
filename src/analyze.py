'''
Xinru Yan
July 2020

Inputs:
    review: a csv file contains amazon review. Must have the following columns: Text, Published, Brand.
            published column must contain a string looks like this: 2014-12-16 00:00:00
    aspect: a string represents the aspect to analyze (e.g Smell).
    key: a txt file contains key words of a specific aspect.
         one key word per line
    year: a string represents the splitting year (e.g. 2014, the program only focuses on data published >= 2014)
    colormap: a txt file contains a list of colors that will be matched to brands for analysis.
              one color per line.
              number of brands should equal to number of colors in the file.
    output_dir: dir name for outputs
'''
import nltk
import os
import click
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Tuple, Dict
import warnings
warnings.simplefilter("ignore")


def set_palette(df: pd.DataFrame, colors: List) -> Dict:
    brands = list(set(df['Brand'].tolist()))
    assert len(brands) == len(colors), f'There are {len(brands)} brands and {len(colors)}, please edit the colormap.txt file so the numbers match.'
    palette = dict(zip(brands, colors))
    return palette


def read_txt(filename: str) -> List[str]:
    lines = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            lines.append(line.strip('\n'))
    return lines


def add_pub_year(df: pd.DataFrame) -> pd.DataFrame:
    df['Published_Year'] = df['Published'].str[:4].astype(int)
    return df


def create_aspect_df(df: pd.DataFrame, keywords: List[str], aspect: str):
    for index, row in df.iterrows():
        text_list = row['Text'].lower().split()
        if len([x for x in text_list if x in keywords]) > 0:
            df.loc[index, aspect] = 1
        else:
            df.loc[index, aspect] = 0
    return df


def data_after_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.loc[(df['Published_Year'] >= year)]
    return df


def split_data(df: pd.DataFrame, aspect) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_1 = df.loc[df[aspect] == 1]
    df_2 = df.loc[df[aspect] == 0]
    return df_1, df_2


def group_split_data_by_brand(df_1: pd.DataFrame, df_2: pd.DataFrame):
    grouped_1 = df_1.groupby('Brand')
    grouped_2 = df_2.groupby('Brand')
    return grouped_1, grouped_2


def aspect_percentage_per_brand(df_1, df_2, aspect: str) -> pd.DataFrame:
    percentage_df = pd.DataFrame()
    i = 0
    for name, group in df_1:
        percentage_df.at[i, 'Brand'] = name
        percentage_df.at[i, f'{aspect}_Percentage'] = group.shape[0] / (group.shape[0] + df_2.get_group(name).shape[0])
        i += 1
    return percentage_df


def plot_percentage(df: pd.DataFrame, aspect: str, output_dir):
    plot = df.plot.bar(x='Brand', y=f'{aspect}_Percentage')
    plt.tight_layout()
    fig = plot.get_figure()
    fig.savefig(f"{output_dir}/Aspect{aspect}_Percentage.png")
    plt.close(fig)


def calculate_overall_aspect_review_percentage(df:pd.DataFrame, aspect: str):
    overall_percentage = pd.DataFrame()
    groupby_years = df.groupby('Published_Year')
    i = 0
    for year, group in groupby_years:
        group_1 = group.loc[group[aspect] == 1]
        group_2 = group.loc[group[aspect] == 0]
        percentage = group_1.shape[0] / group_2.shape[0]
        overall_percentage.at[i, 'Published_Year'] = year
        overall_percentage.at[i, f'{aspect}_Percentage'] = percentage
        i += 1
    return overall_percentage


def plot_overall_percentage_trend(df: pd.DataFrame, aspect: str, output_dir):
    plot = sns.lineplot(x='Published_Year', y=f'{aspect}_Percentage', markers=True, data=df).set_title(f'Aspect{aspect}_Overall_Percentage')
    # plt.ylim(0, 1)
    fig = plot.get_figure()
    fig.savefig(f'{output_dir}/Aspect{aspect}_Overall_Percentage.png', bbox_inches='tight')
    plt.close(fig)


def calculate_yearly_percentage_per_brand(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    yearly_percentage_df = pd.DataFrame()
    groupby_years = df.groupby('Published_Year')
    i = 0
    for year, group in groupby_years:
        groupby_brands = group.groupby('Brand')
        for brand, brand_group in groupby_brands:
            yearly_percentage_df.at[i, 'Published_Year'] = year
            yearly_percentage_df.at[i, 'Brand'] = brand
            group_1 = brand_group.loc[brand_group[aspect] == 1]
            group_2 = brand_group.loc[brand_group[aspect] == 0]
            yearly_percentage_df.at[i, f'{aspect}_percentage'] = group_1.shape[0] / (
                        group_2.shape[0] + group_1.shape[0])
            i += 1
    yearly_percentage_df = yearly_percentage_df.astype({'Published_Year': 'int32'})
    return yearly_percentage_df


def plot_brand_percentage_trend(df, palette, aspect, output_dir):
    plot = sns.lineplot(x='Published_Year', y=f'{aspect}_percentage', hue='Brand', markers=True, data=df,
                 palette=palette).set_title(f'Aspect{aspect}_Percentage_Over_Time')
    fig = plot.get_figure()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig(f'{output_dir}/Aspect{aspect}_Percentage_Over_Time.png', bbox_inches='tight')
    plt.close(fig)


def extract_keyword_sents(df, keywords, col_name):
    df[col_name] = np.empty((len(df), 0)).tolist()
    df = df.astype(object)
    for index, row in df.iterrows():
        keyword_sents = []
        tokenized_sents = sent_tokenize(row['Text'].lower())
        for sent in tokenized_sents:
            if len([x for x in sent.split() if x in keywords]) > 0:
                keyword_sents.append(sent)
        df.at[index, col_name] = keyword_sents
    return df


def df_sa(df, in_col_name, out_col_mean, out_col_std, out_col_num_of_sent):
    sid = SentimentIntensityAnalyzer()
    for index, row in df.iterrows():
        score = []
        for sent in row[in_col_name]:
            ss = sid.polarity_scores(sent)['compound']
            score.append(ss)
        df.at[index, out_col_mean] = np.mean(score)
        df.at[index, out_col_std] = np.std(score)
        df.at[index, out_col_num_of_sent] = len(score)
    return df


def calculate_year_mean(df, aspect):
    groupby_brand = df.groupby('Brand')
    return_df = pd.DataFrame(columns=['Published_Year','MeanSentimentScore', 'StdSentimentScore'])
    for name, group in groupby_brand:
        newdata = []
        for year in set(group['Published_Year'].tolist()):
            newdata.append((year, group.loc[group['Published_Year'] == year][f'{aspect}_SA_Mean_Score'].mean(), group.loc[group['Published_Year'] == year][f'{aspect}_SA_Mean_Score'].std()))
        newdf = pd.DataFrame(newdata, columns=['Published_Year','MeanSentimentScore', 'StdSentimentScore'])
        newdf = newdf.sort_values('Published_Year')
        newdf['Brand'] = name
        return_df = return_df.append(newdf, ignore_index=True)
    return_df = return_df.fillna(0)
    return_df = return_df.astype({'Published_Year': 'int32', 'MeanSentimentScore': 'float32', 'StdSentimentScore': 'float32'})
    return return_df


def plot_average_sentiment(df, out_dir):
    for name, brand in df.groupby('Brand'):
        fig, ax = plt.subplots()
        ax.plot(brand.Published_Year, brand.MeanSentimentScore)
        ax.set_xticks(sorted(brand['Published_Year'].tolist()))
        ax.set(title=f'Brand {name}')
        ax.fill_between(brand.Published_Year, brand.MeanSentimentScore - brand.StdSentimentScore, brand.MeanSentimentScore + brand.StdSentimentScore, alpha=0.25)
        fig.savefig(f'{out_dir}/Brand_{name}_Average_Sentiment_Trend.png')
        plt.close(fig)


def plot_num_of_reviews_per_brand(df, out_dir):
    df = df.astype({'Published_Year': 'int32'})
    groupby_brand = df.groupby('Brand')
    for name, group in groupby_brand:
        plot = group['Published_Year'].value_counts().sort_index().plot.bar(title=f'{out_dir}/{name}_NumOfReviews_Trend')
        fig = plot.get_figure()
        fig.savefig(f'{out_dir}/{name}_NumOfReviews_Trend.png', bbox_inches='tight')
        plt.close(fig)


def splitDataFrameList(df,target_column):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args =(new_rows,target_column))
    new_df = pd.DataFrame(new_rows)
    return new_df


def sents_sa(df, aspect):
    out_col = 'Sent_Score'
    sid = SentimentIntensityAnalyzer()
    polarity = f'{out_col}_Polarity'
    for index, row in df.iterrows():
        ss = sid.polarity_scores(row[f'{aspect}_Sents'])['compound']
        df.at[index, out_col] = ss
        if ss > 0.05:
            df.at[index, polarity] = 1
        elif ss < -0.05:
            df.at[index, polarity] = -1
        else:
            df.at[index, polarity] = 0
    df[polarity] = df[polarity].astype(int)
    return df


def creat_counts_df(df, category):
    df = df.loc[df['Sent_Score_Polarity'] == category]
    newdf = pd.DataFrame()
    for brand in set(df['Brand'].tolist()):
        d = pd.DataFrame.from_dict(df.loc[df['Brand'] == brand]['Published_Year'].value_counts().sort_index().to_dict(), orient='index', columns=['Counts'])
        d['Brand'] = brand
        newdf = newdf.append(d)
    newdf.reset_index(inplace=True)
    newdf = newdf.rename(columns={'index': 'Published_Year'})
    newdf = newdf.astype({'Published_Year': 'int32'})
    return newdf


def create_percentage_df(df1, df2, df3):
    for index, row in df1.iterrows():
        year = row['Published_Year']
        brand = row['Brand']
        if df2.loc[(df2['Brand'] == brand) & (df2['Published_Year'] == year)]['Counts'].empty:
            count2 = 0
        else:
            count2 = df2.loc[(df2['Brand'] == brand) & (df2['Published_Year'] == year)]['Counts'].tolist()[0]
        if df3.loc[(df3['Brand'] == brand) & (df3['Published_Year'] == year)]['Counts'].empty:
            count3 = 0
        else:
            count3 = df3.loc[(df3['Brand'] == brand) & (df3['Published_Year'] == year)]['Counts'].tolist()[0]
        total_counts = row['Counts'] + count2 + count3
        df1.at[index, 'Percentage'] = row['Counts'] / total_counts
    return df1


def plot_fine_grained_sentiment(df, palette, polarity: str, aspect: str, out_dir):
    plot = sns.lineplot(x='Published_Year', y='Percentage', hue='Brand', markers=True, data=df,
                 palette=palette).set_title(f'{polarity}_Trend')
    fig = plot.get_figure()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig(f'{out_dir}/Aspect{aspect}_{polarity}_Percentage_Over_Time', bbox_inches='tight')
    plt.close(fig)


@click.command()
@click.option('-r', '--review', type=str, default='../data/sample_data.csv')
@click.option('-s', '--aspect', type=str, default='Smell')
@click.option('-k', '--key', type=str, default='../data/keywords.txt')
@click.option('-y', '--year', type=int, default=2014)
@click.option('-c', '--colormap', type=str, default='../data/colormap.txt')
@click.option('-o', '--output_dir', type=str, default='../graphs/')
def main(review, aspect, key, year, colormap, output_dir):
    review = pd.read_csv(review)
    keywords = read_txt(key)
    colors = read_txt(colormap)
    output_dir = output_dir + aspect
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_review = data_after_year(create_aspect_df(add_pub_year(review), keywords, aspect), year)
    palette = set_palette(new_review, colors)

    aspect_data, non_aspect_data = split_data(new_review, aspect)

    #brand aspect review percentage
    asepct_groupped_data, non_aspect_groupped_data = group_split_data_by_brand(aspect_data, non_aspect_data)
    percentage_df = aspect_percentage_per_brand(asepct_groupped_data, non_aspect_groupped_data, aspect)
    plot_percentage(percentage_df, aspect, output_dir)

    #overall percentage trend
    overall_percentage = calculate_overall_aspect_review_percentage(new_review, aspect)
    plot_overall_percentage_trend(overall_percentage, aspect, output_dir)

    #brand percentage trend
    yearly_percentage_per_brand = calculate_yearly_percentage_per_brand(new_review, aspect)
    plot_brand_percentage_trend(yearly_percentage_per_brand, palette=palette, aspect=aspect, output_dir=output_dir)

    #brand num of reviews trend bar
    plot_num_of_reviews_per_brand(aspect_data, output_dir)

    #corse grained sentiment analysis
    keyword_sents_df = extract_keyword_sents(df=aspect_data, keywords=keywords, col_name=f'{aspect}_Sents')
    sa_score_df = df_sa(keyword_sents_df, f'{aspect}_Sents', f'{aspect}_SA_Mean_Score', f'{aspect}_SA_Std_Score', f'{aspect}_SA_NumSents')
    score_df = calculate_year_mean(sa_score_df, aspect)
    plot_average_sentiment(score_df, output_dir)


    #fine grained sentiment analysis
    new_sents_df = splitDataFrameList(keyword_sents_df, f'{aspect}_Sents').drop(
        columns=[f'{aspect}_SA_Mean_Score', f'{aspect}_SA_Std_Score', f'{aspect}_SA_NumSents'])
    all_sents = sents_sa(new_sents_df, aspect)

    pos = creat_counts_df(all_sents, 1)
    neg = creat_counts_df(all_sents, -1)
    neu = creat_counts_df(all_sents, 0)

    pos_percent = create_percentage_df(pos, neg, neu)
    neg_percent = create_percentage_df(neg, pos, neu)
    neu_percent = create_percentage_df(neu, neg, pos)

    plot_fine_grained_sentiment(pos_percent, palette=palette, aspect=aspect, polarity='Pos', out_dir=output_dir)
    plot_fine_grained_sentiment(neu_percent, palette=palette, aspect=aspect, polarity='Neu', out_dir=output_dir)
    plot_fine_grained_sentiment(neg_percent, palette=palette, aspect=aspect, polarity='Neg', out_dir=output_dir)


if __name__ == '__main__':
    main()