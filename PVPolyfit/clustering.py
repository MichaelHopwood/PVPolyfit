from PVPolyfit import kernel

from numpy import linalg, zeros, ones, hstack, asarray, vstack, array, mean, std
import pandas as pd

def cluster_ordered_pairs_and_return_df_of_days_in_cluster(cut_results, test_cut_results, ordered_pair_list, test_ordered_pair_list, kmeans_num_clusters = 4, print_info = False):
    ''' KNN Clustering Algorithm - to find the same types of days'''
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters = kmeans_num_clusters)
    kmeans.fit(array(ordered_pair_list + test_ordered_pair_list))

    centroids = kmeans.cluster_centers_
    km_labels = kmeans.labels_
    #print('labels, centroids', km_labels, centroids)

    train_km_labels = km_labels[:len(ordered_pair_list)]
    test_km_labels = km_labels[len(ordered_pair_list):]

    # For cut_results, make new column with each cluster the day belongs to
    for i in range(len(cut_results)):
        cut_results[i]['model_num'] = train_km_labels[i]
    for i in range(len(test_cut_results)):
        test_cut_results[i]['model_num'] = test_km_labels[i]

    train_array_km_labels = array(train_km_labels)
    test_array_km_labels = array(test_km_labels)

    #for i in range(kmeans_num_clusters):
    #    print("[train]: cluster num ", i, " has ", len(train_array_km_labels[train_array_km_labels == i]))

    #for i in range(kmeans_num_clusters):
    #    print("[test ]: cluster num ", i, " has ", len(test_array_km_labels[test_array_km_labels == i]))

    #print("LEN(train_cut_results): ", len(cut_results))
    train_kmeans_dfs = [pd.DataFrame() for i in range(kmeans_num_clusters)]
    train_model_day_count = [0] * kmeans_num_clusters
    for i in range(len(cut_results)):
        train_model_day_count[train_km_labels[i]] += 1
        train_kmeans_dfs[train_km_labels[i]] = pd.concat([train_kmeans_dfs[train_km_labels[i]], cut_results[i]])

    if print_info:
        print("[TRAIN]: NUM DAYS PER MODEL", train_model_day_count)

    test_kmeans_dfs = [pd.DataFrame() for i in range(kmeans_num_clusters)]
    test_model_day_count = [0] * kmeans_num_clusters
    for i in range(len(test_cut_results)):
        test_model_day_count[test_km_labels[i]] += 1
        test_kmeans_dfs[test_km_labels[i]] = pd.concat([test_kmeans_dfs[test_km_labels[i]], test_cut_results[i]])

    if print_info:
        print("[TEST]: NUM DAYS PER MODEL", test_model_day_count)
    #print("LENGTH list of dfs: ", len(test_kmeans_dfs), len(test_km_labels))

    return train_kmeans_dfs, test_kmeans_dfs, test_km_labels, cut_results, test_cut_results, train_model_day_count, test_model_day_count

def save_model_for_each_cluster(kmeans_dfs, degree, Y_tag, xs, kernel_type):
    # each df corresponds to each model

    # save a model for each cluster of days
    saved_models = []
    #print"start of model")
    for i in range(len(kmeans_dfs)):
        #print"len(kmeans_dfs) for model: ", i, " is ", len(kmeans_dfs[i]))
        if len(kmeans_dfs[i]) == 0:
            # if model has no data
            saved_models.append(0)
            continue

        Y = array(kmeans_dfs[i][Y_tag].tolist())

        tup_list = []
        for j in range(len(xs)):
            tup_list.append(array(kmeans_dfs[i][xs[j]].tolist()))

        tup = tuple(tup_list)

        model = kernel.Model(tup, Y, degree, kernel_type)
        
        model.build()

        saved_models.append(model)

    return saved_models

def create_conglomerated_vectors_for_clustering_algorithm(cut_results, hours_kpi, day_hour_list, Y_tag):
    freq_list = []
    # ordered pairs used for kmeans
    ordered_pair_list = []
    big_df = pd.DataFrame(columns = ['variable', 'cloudy', 'slightly cloudy', 'clear', 'hours'])
    dict_list = []
    for l in range(len(cut_results)):
        df = cut_results[l]

        # possible other kpi
        max_v = mean(df[Y_tag].tolist())

        onez, twos, threes, fours = len(df[df['day_type'] == 1]), len(df[df['day_type'] == 2]), len(df[df['day_type'] == 3]), len(df[df['day_type'] == 4])
        if (onez + twos + threes + fours != len(cut_results[l])):
            raise Exception("Sum of classes does not equal length of df")

        # consider adding index here, or the time of day
        freq_df = pd.DataFrame({'variable':onez, 'cloudy':twos, 'slightly cloudy':threes, 'clear':fours}, index = day_hour_list)
        freq_list.append(freq_df)

        ordered_pair = [hours_kpi[l], array(df[Y_tag].tolist()).sum()]
        ordered_pair_list.append(ordered_pair)

        dict = {'variable':onez, 'cloudy':twos, 'slightly cloudy':threes, 'clear':fours, 'hours':hours_kpi[l], 'mean_power':max_v, 'energy':array(df[Y_tag].tolist()).sum()}
        dict_list.append(dict)

    #print("length of dict list and day hour list: ", len(dict_list), len(day_hour_list))
    day_list = list(set(day_hour_list))

    big_df = pd.DataFrame(dict_list, index = day_list)

    #sns.pairplot(big_df)
    #plt.show()
    #print(ordered_pair_list)

    return ordered_pair_list, big_df
