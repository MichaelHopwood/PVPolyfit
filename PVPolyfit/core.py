# Standard
import copy
import warnings
from datetime import datetime
from math import sqrt

# Third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PVPolyfit import clustering as cluster

# Source
from PVPolyfit import kernel
from PVPolyfit import preprocessing as preprocess
from PVPolyfit import utilities
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


def pvpolyfit(
    train_df,
    test_df,
    Y_tag,
    xs,
    I_tag,
    ghi_tag,
    cs_tag,
    highest_num_clusters,
    highest_degree,
    kernel_type,
    Y_high_filter,
    min_count_per_day,
    include_preprocess=False,
    plot_graph=True,
    graph_type="regression",
    print_info=False,
):
    # Sanitation checks
    if len(train_df.index) == 0:
        raise Exception("Training dataframe is empty.")

    if len(test_df.index) == 0:
        raise Exception("Test dataframe is empty.")

    pvpoly = PVPolyfit(train_df, test_df, Y_tag, xs, I_tag, ghi_tag, cs_tag, print_info)

    pvpoly.prepare(Y_high_filter, min_count_per_day, include_preprocess)

    rmse_list = []
    std_rmse_list = []
    pvpoly_objects = []
    combined_labels = []
    for i in range(1, highest_num_clusters + 1):
        pvpoly_iter = copy.deepcopy(pvpoly)
        # Seperate into multiple try/excepts to localize problem
        try:
            labels = pvpoly_iter.run(
                num_clusters=i,
                num_iterations=1,
                degrees=list(range(1, highest_degree + 1)),
                kernel_type=kernel_type,
            )
        except Exception as e:
            if print_info:
                print(e)
            break

        try:
            all_best_dfs, ultimate_days, avg_rmse, std_rmse = pvpoly_iter.evaluate(
                print_info=print_info
            )
        except Exception as e:
            if print_info:
                print(e)
            break

        rmse_list.append(avg_rmse)
        std_rmse_list.append(std_rmse)
        pvpoly_objects.append(pvpoly_iter)
        combined_labels.append(labels)

    if not rmse_list:
        raise Exception("RMSE List was empty -> No Output was produced.")

    min_idx = np.argmin(rmse_list)

    clusters_used = range(1, highest_num_clusters + 1)[min_idx]

    if print_info:
        print("Min idx: {}".format(min_idx))
        print("{} cluster(s) were used.".format(clusters_used))

    days_rmses, model_output, meases, df = pvpoly_objects[min_idx].plot(
        graph_type=graph_type, print_info=print_info, plot_graph=plot_graph
    )

    return (
        model_output,
        meases,
        days_rmses,
        clusters_used,
        df,
        combined_labels[min_idx],
    )


def _pvpolyfit_inputCluster(
    train_df,
    test_df,
    Y_tag,
    xs,
    I_tag,
    ghi_tag,
    cs_tag,
    num_clusters,
    highest_degree,
    kernel_type,
    Y_high_filter,
    min_count_per_day,
    include_preprocess=False,
    plot_graph=True,
    graph_type="regression",
    print_info=False,
):
    # Sanitation checks
    if len(train_df.index) == 0:
        raise Exception("Training dataframe is empty.")

    if len(test_df.index) == 0:
        raise Exception("Test dataframe is empty.")

    pvpoly = PVPolyfit(train_df, test_df, Y_tag, xs, I_tag, ghi_tag, cs_tag, print_info)

    pvpoly.prepare(Y_high_filter, min_count_per_day, include_preprocess)

    try:
        pvpoly.run(
            num_clusters=num_clusters,
            num_iterations=1,
            degrees=list(range(1, highest_degree + 1)),
            kernel_type=kernel_type,
        )
    except Exception as e:
        if print_info:
            print(e)

    try:
        all_best_dfs, ultimate_days, avg_rmse, std_rmse = pvpoly.evaluate(print_info=print_info)
    except Exception as e:
        if print_info:
            print(e)

    if not avg_rmse:
        raise Exception("No Output was produced. Go here for more information: ")

    days_rmses, model_output, meases, df = pvpoly.plot(
        graph_type=graph_type, print_info=print_info, plot_graph=plot_graph
    )

    return model_output, meases, days_rmses, num_clusters, df


def break_days(df, filter_bool, min_count_per_day=8, frequency="days", print_info=False):
    index_list = []
    day_hour_list = []
    prev = 0
    for index, j in enumerate(df.index):
        if not isinstance(j, str):
            print("Index value {} is of type {} with value {}".format(j, type(j), df.loc[j]))
            j = j.strftime("%m/%d/%Y %H:%M:%S %p")
        if frequency == "days":
            curr = int(datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p").strftime("%d"))
            frq = datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p").strftime("%m/%d/%Y")
        elif frequency == "hours":
            curr = int(datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p").strftime("%H"))
            frq = datetime.strptime(j, "%m/%d/%Y %H:%M:%S %p").strftime("%m/%d/%Y %H")
        if curr != prev:
            index_list.append(index)
            day_hour_list.append(frq)
        prev = curr

    cut_results = []

    # Break df into days
    for k in range(len(index_list)):
        if k == (len(index_list) - 1):
            # append lasfinal_df.iloc[[iindex]].indext day
            cut_results.append(df[index_list[k] : -1])
        else:
            cut_results.append(df[index_list[k] : index_list[k + 1]])

    cut_results[-1] = pd.concat([cut_results[-1], df.iloc[[-1]]])

    return index_list, day_hour_list, cut_results, df


def heat_plot(df, N):
    # Nth column of DF will be plotted
    # Inspiration of this code was gathered from Solar-Data-Tools

    index_list, _, cut_df, _ = break_days(df, False)
    lizt = []

    comb_df = pd.DataFrame()

    cut_df = cut_df[1:-1]
    dates = []
    for i in range(len(cut_df)):
        try:
            comb_df[str(i)] = cut_df[i][cut_df[i].columns[N]].tolist()
            dates.append(
                datetime.strptime(cut_df[i].index[0], "%m/%d/%Y %I:%M:%S %p").strftime("%m/%d/%Y")
            )
        except ValueError:
            continue

    lizt = comb_df.values
    fig, ax = plt.subplots(nrows=1, figsize=(10, 8))
    foo = ax.imshow(lizt, cmap="hot", interpolation="none", aspect="auto", vmin=0)

    if df.columns[N] == "error":
        ax.set_title("PVPolyfit Error Heat Plot")

    if df.columns[N] == "rmse":
        ax.set_title("PVPolyfit RMSE Heat Plot")

    if df.columns[N] == "model_output":
        ax.set_title("PVPolyfit Model Output Heat Plot")

    plt.colorbar(foo, ax=ax, label="W")
    ax.set_xlabel("Day number")
    ax.set_xticks(np.arange(len(dates)))
    ax.set_xticklabels(dates)

    ax.set_yticks([])
    ax.set_ylabel("              Time of day                 ")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    plt.show()


class PVPolyfit:
    """
    .______   ____    ____ .______     ______    __      ____    ____  _______  __  .___________.
    |   _  \  \   \  /   / |   _  \   /  __  \  |  |     \   \  /   / |   ____||  | |           |
    |  |_)  |  \   \/   /  |  |_)  | |  |  |  | |  |      \   \/   /  |  |__   |  | `---|  |----`
    |   ___/    \      /   |   ___/  |  |  |  | |  |       \_    _/   |   __|  |  |     |  |
    |  |         \    /    |  |      |  `--'  | |  `----.    |  |     |  |     |  |     |  |
    | _|          \__/     | _|       \______/  |_______|    |__|     |__|     |__|     |__|

    An object, PVPolyfit, created for the creation of an accurate regression of Output depending on the two covariates

    PARAMETERS
    ----------

        train_df: df
            holds training data with columns and index specified below
        test_df: df
            holds testing data with columns and index specified below
        Y_tag: str
            column name of output tag
        xs: list of str
            list of column names for two covariates
        ghi_tag: str
            column name of GHI input
        cs_tag: str
            column name of clearsky GHI generated by pvlib simulation (link below)

    USER MUST INPUT DF's WITH FOLLOWING COLUMNS:

    |       Description      |  Original Use Case  |   Model Purpose          |
    |------------------------|---------------------|--------------------------|
    | Output,       Y_tag:   | DC Power            | Target for regression    |
    | xs:              x1:   | POA Irradiance      | Covariate for regression |
    |                  x2:   | Ambient Temperature | Covariate for regression |
    | Measured GHI, ghi_tag  | GHI (irradiance)    | Day classification       |
    | PVLib Clearsky, cs_tag | Simulated GHI       | Day classification       |

    PVLib has a good tutorial to generate clearsky data:
        https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.location.Location.get_clearsky.html
    """

    def __init__(self, train_df, test_df, Y_tag, xs, I_tag, ghi_tag, cs_tag, print_info):
        self.train_df = train_df
        self.test_df = test_df
        self.Y_tag = Y_tag
        self.xs = xs
        self.I_tag = I_tag
        self.ghi_tag = ghi_tag
        self.cs_tag = cs_tag
        self.print_info = print_info

        self.num_clusters = 0
        self.num_iterations = 0
        self.degrees = []

        self.cut_results = []
        self.test_cut_results = []
        self.ordered_pair_list = []
        self.test_ordered_pair_list = []
        self.combined_test_cut_results = []
        self.test_km_labels = []
        self.avg_rmse = 0

        # all_best_dfs[Degree][Day][Columns: 'Y', 'mins', 'maxs']
        self.all_best_dfs = []

        # model_day_couts[Degree][Best model's [Train Counts, Test Counts]]
        self.model_day_counts = []

        # ultimate_days[Day][i, ind]
        self.ultimate_days = []

    def prepare(self, Y_high_filter, min_count_per_day, include_preprocess, use_mhopwood=False):
        """ Preprocess and classify days in DataFrame """
        self.train_df = preprocess.data_preprocessing(
            self.train_df,
            self.xs,
            self.Y_tag,
            self.I_tag,
            self.cs_tag,
            Y_high_filter,
            self.print_info,
            include_preprocess,
        )
        if use_mhopwood:
            if include_preprocess:
                classification = cluster.classify_weather_day_MHopwood(
                    self.cut_results, self.Y_tag, self.xs, kmeans_num_clusters=4
                )
                self.train_df["day_type"] = classification
        else:
            if include_preprocess:
                classification, _, _ = preprocess.classify_weather_day_GM_Tina(
                    self.train_df, self.cs_tag, self.ghi_tag
                )
                self.train_df["day_type"] = classification

        # cuts train_df into daily DF's
        # also returns a filtered train_df which cuts out a day if its length is too small
        (
            index_list,
            day_hour_list,
            self.cut_results,
            self.train_df,
        ) = utilities.find_and_break_days_or_hours(
            self.train_df,
            True,
            min_count_per_day=min_count_per_day,
            frequency="days",
            print_info=self.print_info,
        )

        (
            middles_dates,
            hours_kpi,
        ) = utilities.get_weighted_middle_of_day_and_calculate_float_since_noon(
            self.cut_results, self.Y_tag
        )
        # For each day, compile frequencies
        # For each day, output the # times each class is triggered

        (
            self.ordered_pair_list,
            freq_df,
        ) = cluster.create_conglomerated_vectors_for_clustering_algorithm(
            self.cut_results, hours_kpi, day_hour_list, self.Y_tag, self.xs
        )

        self.test_df = preprocess.data_preprocessing(
            self.test_df,
            self.xs,
            self.Y_tag,
            self.I_tag,
            self.cs_tag,
            Y_high_filter,
            self.print_info,
            include_preprocess,
        )

        if len(self.cs_tag) != 0 or len(self.ghi_tag) != 0:
            test_classification, _, _ = preprocess.classify_weather_day_GM_Tina(
                self.test_df, self.cs_tag, self.ghi_tag
            )
            self.test_df["day_type"] = test_classification

        (
            _,
            test_day_hour_list,
            self.test_cut_results,
            self.test_df,
        ) = utilities.find_and_break_days_or_hours(
            self.test_df,
            True,
            min_count_per_day=min_count_per_day,
            frequency="days",
            print_info=self.print_info,
        )

        _, test_hours_kpi = utilities.get_weighted_middle_of_day_and_calculate_float_since_noon(
            self.test_cut_results, self.Y_tag
        )

        print("TEST CUT RESULTS CREATE CONGLOMERATED")
        (
            self.test_ordered_pair_list,
            test_freq_df,
        ) = cluster.create_conglomerated_vectors_for_clustering_algorithm(
            self.test_cut_results, test_hours_kpi, test_day_hour_list, self.Y_tag, self.xs
        )

    def run(self, num_clusters=6, num_iterations=1, degrees=None, kernel_type="polynomial"):
        """
        Iterates through Degrees
        For each Degree, iterates n times
        Returns best model for each input day

        Parameters:
            num_clusters: int, default 6
                number of clusters used in clustering algorithm, synonymous with number of 'types of days'

            num_iterations: int, default 1
                number of times algorithm loops, indicates volatility of algorithm (usually very small, so default = 1)

            degrees: list of ints
                range of degrees that polynomial kernel iterates through

            kernel_type: str
                type of regression kernel to be used
                OPTIONS: polynomial - a(AB)+
        """
        if degrees is None:
            degrees = list(range(1, 10))

        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.degrees = degrees
        self.kernel_type = kernel_type

        self.all_best_dfs = []
        self.model_day_counts = []

        for degree in self.degrees:
            P_se_list = []
            combined_P_list = []
            combined_day_counts = []
            combined_test_km_labels = []

            # 1. Run the code an n number of times
            for _ in range(self.num_iterations):
                # clusters and adds 'model_num' column to cut_results & test_cut_results
                (
                    train_kmeans_dfs,
                    test_kmeans_dfs,
                    self.test_km_labels,
                    self.cut_results,
                    self.test_cut_results,
                    train_model_day_count,
                    test_model_day_count,
                ) = cluster.cluster_ordered_pairs_and_return_df_of_days_in_cluster(
                    self.cut_results,
                    self.test_cut_results,
                    self.ordered_pair_list,
                    self.test_ordered_pair_list,
                    kmeans_num_clusters=self.num_clusters,
                    print_info=self.print_info,
                )

                saved_models = cluster.save_model_for_each_cluster(
                    train_kmeans_dfs, degree, self.Y_tag, self.xs, self.kernel_type
                )

                self.kmeans_Y_lists = kernel.process_test_data_through_models(
                    test_kmeans_dfs, saved_models, self.test_km_labels, self.xs
                )

                # 2. For each iteration, save the modelled P and colors (based on model used)
                combined_P_list.append(self.kmeans_Y_lists)
                self.combined_test_cut_results.append(self.test_cut_results)
                combined_test_km_labels.append(self.test_km_labels)
                combined_day_counts.append([train_model_day_count, test_model_day_count])
                P_se_km = kernel.EvaluateModel(
                    self.test_df[self.Y_tag].values, np.array(self.kmeans_Y_lists)
                ).rmse()
                P_se_list.append(P_se_km)

            # 3. Gather the minimum and maximum for each index, save in two lists
            mins = []
            maxs = []
            for i in range(len(self.test_df.index)):
                _min = 9999
                _max = -9999
                for j in range(len(combined_P_list)):
                    if combined_P_list[j][i] < _min:
                        _min = combined_P_list[j][i]
                    if combined_P_list[j][i] > _max:
                        _max = combined_P_list[j][i]
                mins.append(_min)
                maxs.append(_max)

            best_index = np.argmin(P_se_list)
            best_model = combined_P_list[best_index]
            best_df = pd.DataFrame()

            best_df["Y"] = best_model
            best_df["mins"] = mins
            best_df["maxs"] = maxs

            best_df.index = self.test_df.index
            _, _, dfg, _ = utilities.find_and_break_days_or_hours(
                best_df, False, min_count_per_day=0, frequency="days", print_info=self.print_info
            )
            self.all_best_dfs.append(dfg)
            self.model_day_counts.append(combined_day_counts[best_index])
            return combined_test_km_labels[best_index]

    def evaluate(self, print_info=True):
        """
            Determine rmse for each day for each degree
            and return index of best model for each day
        """

        # iterate by day
        all_rmse = []
        self.ultimate_days = []
        for i in range(len(self.all_best_dfs[0])):
            _min = 9999
            ind = 0
            # iterate by degree
            for j in range(len(self.all_best_dfs)):
                iterating_rmse = kernel.EvaluateModel(
                    self.test_cut_results[i][self.Y_tag].values,
                    self.all_best_dfs[j][i]["Y"].values,
                ).rmse()
                print("Degree ", j, " has error: ", iterating_rmse)
                if abs(iterating_rmse) < abs(_min):
                    _min = iterating_rmse
                    ind = j
            if print_info:
                print(
                    "{} index: {}, degrees len: {}".format(
                        len(self.all_best_dfs), ind, len(self.degrees)
                    )
                )
                print("Day {} chooses degree {} with {}".format(i, self.degrees[ind], _min))
            all_rmse.append(_min)
            self.ultimate_days.append([i, ind])

        self.avg_rmse = np.mean(all_rmse)
        return self.all_best_dfs, self.ultimate_days, self.avg_rmse, np.std(all_rmse)

    def plot(self, graph_type="regression", print_info=True, plot_graph=False):
        # Initialize at start so no error thrown if plot is not `regression`
        iter_rmses = []
        model_outputs = []
        meases = []
        df = pd.DataFrame(columns=["error", "model_output", "meas"])

        if graph_type == "regression":
            colors = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "gold",
                "pink",
                "gray",
                "cyan",
                "darkgreen",
                "cadetblue",
                "lawngreen",
                "cornflowerblue",
                "navy",
                "olive",
                "orangered",
                "orchid",
                "plum",
                "khaki",
                "ivory",
                "magenta",
                "maroon",
                "plum",
                "cyan",
                "crimson",
                "coral",
                "yellowgreen",
                "wheat",
                "sienna",
                "salmon",
            ] * 5
            df_index = []
            uncer_vals = []
            df_meases = []
            for i in range(len(self.all_best_dfs[0])):
                model_number = self.test_km_labels[i]
                color = colors[model_number]

                ind = self.ultimate_days[i][1]

                Y_output_daily = self.all_best_dfs[ind][i]["Y"].tolist()
                model_outputs.append(Y_output_daily)
                day_index = self.all_best_dfs[ind][i].index.tolist()
                day_maxes = self.all_best_dfs[ind][i]["maxs"].tolist()
                day_mins = self.all_best_dfs[ind][i]["mins"].tolist()
                day_meas = self.test_cut_results[i][self.Y_tag].values
                meases.append(day_meas)
                dt_index = pd.to_datetime(day_index)

                if plot_graph:
                    plt.plot(dt_index, day_meas, "k")
                    plt.plot(dt_index, Y_output_daily, color)
                    plt.fill_between(dt_index, day_maxes, day_mins, facecolor=color)
                    plt.xlabel("time")
                    plt.ylabel("Watts")
                    plt.xticks(rotation=60)
                    plt.title("Modelled Multiple Day Types (by color)")

                uncer = np.array(Y_output_daily) - day_meas  # /(day_meas))
                calc_rmse = sqrt(mean_squared_error(day_meas, np.array(Y_output_daily)))
                iter_rmses.append(calc_rmse)
                df_index.append(dt_index)
                uncer_vals.append(uncer)
                df_meases.append(day_meas)

                if print_info:
                    print(
                        "[{}]:".format(
                            datetime.strptime(day_index[0], "%m/%d/%Y %H:%M:%S %p").strftime(
                                "%Y-%m-%d"
                            )
                        )
                    )
                    print("\trmse: {:.4f}, error: {:.4f}".format(calc_rmse, uncer.mean()))
            if plot_graph:
                plt.show()

            plt.close()

            uncer_values = [item for sublist in uncer_vals for item in sublist]
            df_indices = [item for sublist in df_index for item in sublist]

            df = pd.DataFrame(index=df_indices)
            df["error"] = uncer_values
            df["model_output"] = [item for sublist in model_outputs for item in sublist]
            df["meas"] = [item for sublist in df_meases for item in sublist]

        return iter_rmses, model_outputs, meases, df

    def plot_classification_map(self):
        return

    def model_information(self):
        return
