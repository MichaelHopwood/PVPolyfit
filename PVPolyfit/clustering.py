# Standard
import datetime

# Third party
import numpy as np
import pandas as pd

# Source
from PVPolyfit import kernel
from sklearn.cluster import KMeans


def classify_weather_day_MHopwood(cut_results, Y_tag, xs, kmeans_num_clusters=4):
    # make parameters for sub-daily
    classifications = []
    for df in cut_results:
        daysecs = []
        for index in df.index:
            # Calculate once
            _datetime = datetime.datetime.strptime(index, "%m/%d/%Y %H:%M:%S %p")
            _daysecs = (
                _datetime
                - _datetime.replace(hour=0, minute=0, second=0, microsecond=0).total_seconds()
            )
            daysecs.append(_daysecs)

        vals_for_vstack = [df[x].values for x in xs]

        ordered_pairs = np.hstack((np.vstack(daysecs), np.vstack(vals_for_vstack).T))

        kmeans_day = KMeans(n_clusters=kmeans_num_clusters)
        kmeans_day.fit(ordered_pairs)

        km_labels = kmeans_day.labels_

        classifications.append(km_labels)

    # flatten classifications
    classification = [item for sublist in classifications for item in sublist]
    print(f"Classification: {classification}")

    return classification


def cluster_ordered_pairs_and_return_df_of_days_in_cluster(
    cut_results,
    test_cut_results,
    ordered_pair_list,
    test_ordered_pair_list,
    kmeans_num_clusters=4,
    print_info=False,
):
    """ KNN Clustering Algorithm - to find the same types of days"""

    kmeans = KMeans(n_clusters=kmeans_num_clusters)
    kmeans.fit(np.array(ordered_pair_list + test_ordered_pair_list))

    km_labels = kmeans.labels_

    train_km_labels = km_labels[: len(ordered_pair_list)]
    test_km_labels = km_labels[len(ordered_pair_list) :]

    # For cut_results, make new column with each cluster the day belongs to
    for cut_result, train_km_label in zip(cut_results, train_km_labels):
        cut_results["model_num"] = train_km_label

    for test_cut_result, test_km_label in zip(test_cut_results, test_km_labels):
        test_cut_result["model_num"] = test_km_label

    train_kmeans_dfs = [pd.DataFrame() for i in range(kmeans_num_clusters)]
    train_model_day_count = [0] * kmeans_num_clusters
    for index, cut_result in enumerate(cut_results):
        train_model_day_count[train_km_labels[index]] += 1
        train_kmeans_dfs[train_km_labels[index]] = pd.concat(
            [train_kmeans_dfs[train_km_labels[index]], cut_result]
        )

    if print_info:
        print("[TRAIN]: NUM DAYS PER MODEL", train_model_day_count)

    test_kmeans_dfs = [pd.DataFrame() for _ in range(kmeans_num_clusters)]
    test_model_day_count = [0] * kmeans_num_clusters
    for index, test_cut_result in enumerate(test_cut_results):
        test_model_day_count[test_km_labels[index]] += 1
        test_kmeans_dfs[test_km_labels[index]] = pd.concat(
            [test_kmeans_dfs[test_km_labels[index]], test_cut_result]
        )

    if print_info:
        print("[TEST]: NUM DAYS PER MODEL", test_model_day_count)

    return (
        train_kmeans_dfs,
        test_kmeans_dfs,
        test_km_labels,
        cut_results,
        test_cut_results,
        train_model_day_count,
        test_model_day_count,
    )


def save_model_for_each_cluster(kmeans_dfs, degree, Y_tag, xs, kernel_type):
    # each df corresponds to each model

    # save a model for each cluster of days
    saved_models = []
    # print"start of model")
    for kmeans_df in kmeans_dfs:

        if len(kmeans_df.index) == 0:
            # if model has no data
            saved_models.append(0)
            continue

        Y = kmeans_df[Y_tag].values

        inputs = [kmeans_df[x].values for x in xs]

        model = kernel.Model(inputs, Y, degree, kernel_type)
        model.build()
        saved_models.append(model)

    return saved_models


def create_conglomerated_vectors_for_clustering_algorithm(
    cut_results, hours_kpi, day_hour_list, Y_tag, xs
):
    normal = "Maybe"
    # normal = True

    big_df = pd.DataFrame()

    if normal == True:  # noqa: ignore
        # ordered pairs used for kmeans
        ordered_pair_list = []
        big_df = pd.DataFrame(columns=["variable", "cloudy", "slightly cloudy", "clear", "hours"])
        df_rows = []
        for index, df in enumerate(cut_results):
            # possible other kpi
            max_v = df[Y_tag].max()

            onez, twos, threes, fours = (
                len(df[df["day_type"] == 1]),
                len(df[df["day_type"] == 2]),
                len(df[df["day_type"] == 3]),
                len(df[df["day_type"] == 4]),
            )

            ordered_pair = [
                hours_kpi[index],
                df[Y_tag].sum(),
                max_v,
                onez,
                twos,
                threes,
                fours,
            ]
            ordered_pair_list.append(ordered_pair)

            df_row = {
                "variable": onez,
                "cloudy": twos,
                "slightly cloudy": threes,
                "clear": fours,
                "hours": hours_kpi[index],
                "mean_power": max_v,
                "energy": df[Y_tag].sum(),
            }
            df_rows.append(df_row)

        day_list = list(set(day_hour_list))

        big_df = pd.DataFrame(df_rows, index=day_list)

    elif normal == "Maybe":
        # ordered pairs used for kmeans
        ordered_pair_list = [
            [
                hours_kpi[index],
                df[Y_tag].max(),
                sum((df[Y_tag][j] - df[Y_tag][j - 1] for j in range(1, len(df)))),
            ]
            for index, df in enumerate(cut_results)
        ]
    elif normal == "New":
        ordered_pair_list = [
            [
                hours_kpi[index],
                df[Y_tag].max(),
                sum((df[Y_tag][j] + df[Y_tag][j - 1] for j in range(1, len(df)))),
            ]
            for index, df in enumerate(cut_results)
        ]
    else:
        # ordered pairs used for kmeans
        ordered_pair_list = [
            [hours_kpi[index], df[Y_tag].sum()] for index, df in enumerate(cut_results)
        ]

    return ordered_pair_list, big_df
