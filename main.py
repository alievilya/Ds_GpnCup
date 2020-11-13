from functions import from_parquet, process_empty_fields, replace_string_columns, normalize_column, \
                    plot_clusters, perform_clustering, select_days, plot_bar


if __name__ == "__main__":
    table = from_parquet()
    table = process_empty_fields(table)
    table, labels_dictionary = replace_string_columns(table)
    table = normalize_column(table, 'total_items_sold')
    tab_res = perform_clustering(table)
    final_table2 = select_days(tab_res)
    final_table = final_table2.head(100)
    x_features = ['shop_type']  # 'neighborhood', 'shop_type', 'neighborhood']
    features_to_discover = ['year_opened']
    numeric_feautures = ['total_items_sold', 'year_opened', 'number_of_counters']  # количественные признаки


    plot_clusters(final_table, labels_dictionary, x_features, numeric_feautures, features_to_discover)

    plot_bar(final_table, feature='shop_id')
