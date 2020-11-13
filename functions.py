import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler


def from_parquet():
    """
    reading data from 3 parquet tables and joining them together using chosen indexes
    returns joined table
    """

    table1 = pd.read_parquet('sales.parquet',
                             columns=['date', 'shop_id', 'owner', 'number_of_counters',
                                      'goods_type', 'total_items_sold', ],
                             engine='pyarrow')

    table2 = pd.read_parquet('shops.parquet',
                             columns=['shop_id', 'neighborhood', 'city',
                                      'year_opened', 'is_on_the_road', 'is_with_the_well',
                                      'is_with_additional_services', 'shop_type', ],
                             engine='pyarrow')

    table3 = pd.read_parquet('cities.parquet', engine='pyarrow')

    table1 = pd.DataFrame(table1)
    table2 = pd.DataFrame(table2)
    table3 = pd.DataFrame(table3)
    table23 = table2.join(table3.set_index('city'), on='city')
    full_table = table23.join(table1.set_index('shop_id'))
    return full_table


def process_empty_fields(table):
    """
    replaces NaN fields with string
    returns filled table
    """

    table = table.fillna({"is_on_the_road": "неизвестно"})
    table = table.fillna({"is_with_the_well": "неизвестно"})
    table = table.fillna({"is_with_additional_services": "неизвестно"})
    table = table.fillna({"shop_type": "неизвестно"})
    table = table.fillna({"location": "неизвестно"})
    table = table.fillna({"city": "неизвестно"})
    table = table.loc[table['year_opened'] != -1]
    #     table = table.replace(-1, value="неизвестно")
    #     table.loc[table['year_opened'] == -1] = 'неизвестно'

    return table


def change_type(table, col_name):
    """
    changes type of input table`s column to a str
    """

    column = table[col_name].copy()
    column = str(column)
    table[col_name] = column
    return table


def replace_string_columns(table):
    """
    takes table with object fields as an input and encodes them to an integer values
    returns table and dictionary with values, which will be needed to decode integer fields
    """
    obj_df = table.select_dtypes(include=['object']).copy()
    new_dict = dict()
    col_names = [col_name for col_name in obj_df]
    col_names.append('date')
    for col in col_names:
        if col == "shop_id":
            continue
        new_dict.update({col: {n: cat for n, cat in enumerate(table[col].astype('category').cat.categories)}})
        table[col] = table[col].astype('category')
        table[col] = table[col].cat.codes
    return table, new_dict


def normalize_column(table, col_name):
    """
    applies min-max normalization to a column: col_name in a table
    """
    column = table[col_name].copy()
    normalized_col = (column - column.min()) / (column.max() - column.min())
    table[col_name] = normalized_col
    return table


def perform_clustering(table):
    """
    performs agglomerative clustering on input table, n_clusters=3
    :param table:
    :return: res_table
    """
    scaler = MinMaxScaler()
    all_columns = [n for n in table]
    train_columns = ['neighborhood', 'city', 'year_opened', 'is_on_the_road',
                     'is_with_the_well', 'is_with_additional_services',
                     'shop_type', 'location', 'number_of_counters',
                     'goods_type', 'total_items_sold']

    res_table = pd.DataFrame(columns=all_columns)
    res_table['cluster'] = None
    max_days = table['date'].nunique()
    for n in range(0, max_days):
        small_table = pd.DataFrame()
        small_table = table.loc[table['date'] == n]
        chosen_table = small_table.loc[:, train_columns]
        transformed_table = scaler.fit_transform(chosen_table)
        clustering = AgglomerativeClustering(linkage="ward", n_clusters=3).fit(transformed_table)
        final = small_table.copy()
        final['cluster'] = clustering.labels_
        res_table = res_table.append(final)

    return res_table


def select_days(table):
    """
    choses the most frequently assigned cluster for each shop_id
    :param table:
    :return: result table
    """
    all_columns = [n for n in table]
    res_table = pd.DataFrame(columns=all_columns)
    res_table['cluster'] = None
    max_shops = table['shop_id'].nunique()
    for n in range(0, max_shops):
        shops_table = pd.DataFrame()
        shops_table = table.loc[table['shop_id'] == n]
        if shops_table.empty:
            continue
        max_probable_cluster = shops_table['cluster'].value_counts().idxmax()
        res = shops_table.copy()
        res['cluster'] = max_probable_cluster
        res_table = res_table.append(res)
    return res_table


def plot_clusters(table, labels_dictionary, x_features, num_feautures, features_to_discover):
    """
    plots different features with respect to assigned clusters
    :param table: input table
    :param labels_dictionary: decoding dictionary
    :param x_features: features will be plotted on axis x
    :param num_feautures: table`s columns, which include numeric values
    :param features_to_discover: features will be plotted on axis y
    :return:
    """
    for x_feature in x_features:
        for y_feature in features_to_discover:
            if y_feature == x_feature:
                continue
            fig, ax = plt.subplots()
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)

            colours = ListedColormap(['crimson', 'darkviolet', 'mediumblue', 'green', 'deepskyblue', 'goldenrod'])
            classes = [x for x in range(3)]

            scatter = plt.scatter(table.loc[:, x_feature], table.loc[:, y_feature], c=table.loc[:, 'cluster'],
                                  s=15, cmap=colours)
            plt.legend(handles=scatter.legend_elements()[0], labels=classes)

            # необходимо декодирование, так как это строковый признак
            if y_feature not in num_feautures:
                all_values = labels_dictionary[y_feature]
                position = np.arange(len(labels_dictionary[y_feature]))
                ax.set_yticks(position)
                values = [all_values[n] for n in all_values]
                labels = ax.set_yticklabels(values,
                                            fontsize=10,  # Размер шрифта
                                            color='rebeccapurple',  # Цвет текста
                                            rotation=0,  # Поворот текста
                                            verticalalignment='center')

            if x_feature not in num_feautures:
                position = np.arange(len(labels_dictionary[x_feature]))
                ax.set_xticks(position)
                all_values = labels_dictionary[x_feature]
                values = [all_values[n] for n in all_values]
                labels = ax.set_xticklabels(values,
                                            fontsize=10,  # Размер шрифта
                                            color='indigo',  # Цвет текста
                                            rotation=80,  # Поворот текста
                                            verticalalignment='top')
            plt.show()


def plot_bar(table, feature='shop_id'):
    # final.groupby(["cluster", "total_items_sold"])["cluster"].size().unstack()
    table.groupby(["cluster", feature])["cluster"].size().unstack()
    temp = table.groupby(["cluster"])["cluster"].count()
    plt.style.use("ggplot")
    plt.figure(figsize=(15, 9))
    plt.barh([0, 1, 2], temp.values, label="Cluster Count", color="b")
    plt.ylabel("Clusters")
    plt.xlabel("Count")
    plt.title("Cluster VS count plot")
    plt.xlim(0, temp.values.max())
    plt.yticks([0, 1, 2], ["Cluster 0", "Cluster 1", "Cluster 2"])
    plt.legend()
    plt.show()
