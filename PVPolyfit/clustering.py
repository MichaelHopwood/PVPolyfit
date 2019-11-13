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
    kmeans_num_clusters: int = 4
    classifications: list = []
    for df in cut_results:
        np.array(df.index.tolist())
        daysecs: list = [
            (
                datetime.datetime.strptime(ind, "%m/%d/%Y %H:%M:%S %p")
                - datetime.datetime.strptime(ind, "%m/%d/%Y %H:%M:%S %p").replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            ).total_seconds()
            for ind, row in df.iterrows()
        ]

        tup_list: list = []
        for j in range(len(xs)):
            tup_list.append(np.array(df[xs[j]].tolist()))

        tup: tuple = tuple(tup_list)

        ordered_pairs: np.ndarray = np.hstack((np.vstack(daysecs), np.vstack(tup).T))

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
    for i in range(len(cut_results)):
        cut_results[i]["model_num"] = train_km_labels[i]

    for i in range(len(test_cut_results)):
        test_cut_results[i]["model_num"] = test_km_labels[i]

    train_kmeans_dfs = [pd.DataFrame() for i in range(kmeans_num_clusters)]
    train_model_day_count = [0] * kmeans_num_clusters
    for i in range(len(cut_results)):
        train_model_day_count[train_km_labels[i]] += 1
        train_kmeans_dfs[train_km_labels[i]] = pd.concat(
            [train_kmeans_dfs[train_km_labels[i]], cut_results[i]]
        )

    if print_info:
        print("[TRAIN]: NUM DAYS PER MODEL", train_model_day_count)

    test_kmeans_dfs = [pd.DataFrame() for i in range(kmeans_num_clusters)]
    test_model_day_count = [0] * kmeans_num_clusters
    for i in range(len(test_cut_results)):
        test_model_day_count[test_km_labels[i]] += 1
        test_kmeans_dfs[test_km_labels[i]] = pd.concat(
            [test_kmeans_dfs[test_km_labels[i]], test_cut_results[i]]
        )

    if print_info:
        print("[TEST]: NUM DAYS PER MODEL", test_model_day_count)
    # print("LENGTH list of dfs: ", len(test_kmeans_dfs), len(test_km_labels))

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
    for i in range(len(kmeans_dfs)):
        # print"len(kmeans_dfs) for model: ", i, " is ", len(kmeans_dfs[i]))
        if len(kmeans_dfs[i]) == 0:
            # if model has no data
            saved_models.append(0)
            continue

        Y = np.array(kmeans_dfs[i][Y_tag].tolist())
        # Y_log = np.log(Y)

        tup_list = []
        for j in range(len(xs)):
            tup_list.append(np.array(kmeans_dfs[i][xs[j]].tolist()))

        tup = tuple(tup_list)
        model = kernel.Model(tup, Y, degree, kernel_type)
        model.build()
        saved_models.append(model)

    return saved_models


def create_conglomerated_vectors_for_clustering_algorithm(
    cut_results, hours_kpi, day_hour_list, Y_tag, xs
):
    normal = "Maybe"
    # normal = True

    if normal == True:
        freq_list = []
        # ordered pairs used for kmeans
        ordered_pair_list = []
        big_df = pd.DataFrame(columns=["variable", "cloudy", "slightly cloudy", "clear", "hours"])
        dict_list = []
        for l in range(len(cut_results)):
            df = cut_results[l]

            # possible other kpi
            max_v = max(df[Y_tag].tolist())

            onez, twos, threes, fours = (
                len(df[df["day_type"] == 1]),
                len(df[df["day_type"] == 2]),
                len(df[df["day_type"] == 3]),
                len(df[df["day_type"] == 4]),
            )

            # consider adding index here, or the time of day
            freq_df = pd.DataFrame(
                {"variable": onez, "cloudy": twos, "slightly cloudy": threes, "clear": fours},
                index=day_hour_list,
            )
            freq_list.append(freq_df)

            ordered_pair = [
                hours_kpi[l],
                np.array(df[Y_tag].tolist()).sum(),
                max_v,
                onez,
                twos,
                threes,
                fours,
            ]
            ordered_pair_list.append(ordered_pair)

            d = {
                "variable": onez,
                "cloudy": twos,
                "slightly cloudy": threes,
                "clear": fours,
                "hours": hours_kpi[l],
                "mean_power": max_v,
                "energy": np.array(df[Y_tag].tolist()).sum(),
            }
            dict_list.append(d)

        day_list = list(set(day_hour_list))

        big_df = pd.DataFrame(dict_list, index=day_list)

    elif normal == "Maybe":
        # ordered pairs used for kmeans
        ordered_pair_list = []

        dict_list = []
        for l in range(len(cut_results)):
            df = cut_results[l]

            # max Y
            max_Y = max(df[Y_tag].tolist())

            # total distance Y
            total_distance_Y = 0
            for j in range(1, len(cut_results[l])):
                total_distance_Y += cut_results[l][Y_tag][j] - cut_results[l][Y_tag][j - 1]

            ordered_pair = [hours_kpi[l], max_Y, total_distance_Y]
            ordered_pair_list.append(ordered_pair)

        big_df = pd.DataFrame()

    elif normal == "New":

        # ordered pairs used for kmeans
        ordered_pair_list = []

        dict_list = []
        for l in range(len(cut_results)):
            df = cut_results[l]

            onez, twos, threes, fours = (
                len(df[df["day_type"] == 0]),
                len(df[df["day_type"] == 1]),
                len(df[df["day_type"] == 2]),
                len(df[df["day_type"] == 3]),
            )

            # max Y
            max_Y = max(df[Y_tag].tolist())

            # total distance Y
            total_distance_Y = 0
            for j in range(1, len(cut_results[l])):
                total_distance_Y += cut_results[l][Y_tag][j] + cut_results[l][Y_tag][j - 1]

            # np.array(df[Y_tag].tolist()).sum()]#[onez, twos, threes, fours,
            # hours_kpi[l], np.array(df[Y_tag].tolist()).sum()]
            ordered_pair = [hours_kpi[l], max_Y, total_distance_Y]
            ordered_pair_list.append(ordered_pair)

    else:
        # ordered pairs used for kmeans
        ordered_pair_list = []
        # big_df = pd.DataFrame(columns = ['variable', 'cloudy', 'slightly cloudy', 'clear', 'hours'])
        dict_list = []
        for l in range(len(cut_results)):
            df = cut_results[l]

            # possible other kpi
            max_v = np.mean(df[Y_tag].tolist())

            # [onez, twos, threes, fours, hours_kpi[l], np.array(df[Y_tag].tolist()).sum()]
            ordered_pair = [hours_kpi[l], np.array(df[Y_tag].tolist()).sum()]
            ordered_pair_list.append(ordered_pair)

        big_df = pd.DataFrame()

    return ordered_pair_list, big_df
