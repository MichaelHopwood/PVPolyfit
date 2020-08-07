[![PyPI version](https://badge.fury.io/py/PVPolyfit.svg)](https://badge.fury.io/for/py/PVPolyfit)
[![Downloads](https://pepy.tech/badge/pvpolyfit)](https://pepy.tech/project/pvpolyfit)
# PVPolyfit

A one-lined function call for a high-resolution regression on PV Output

### Overview

PVPolyfit provides a purely mathematical approach towards regression on a physical solar field. With PVPolyfit, you can regress on DC Power using only Irradiance and Ambient Temperature. Further research will need to be conducted to find other applications of this algorithm.

The PVPolyfit algorithm is:
* `FAST` - simulating months of data in minutes

* `ROBUST` - working on a vast variety of datasets and levels - module, string, inverter

* `MALLEABLE` - allowing any number of covariates

* `ACCURATE` - generating one percent error or less

* `USER FRIENDLY` - requiring only one line of code

### Getting it
To download PVPolyfit, either fork this github repo or simply call the following command in your command prompt.
```sh
$ pip install PVPolyfit
```

### Using it

Simply, call the following function located in PVPolyit.core

```py
>>> from PVPolyfit.core import pvpolyfit
>>> modelled_Y, days_rmses = pvpolyfit(train_df, test_df, Y_tag, xs, I_tag, ghi_tag, cs_tag, 
	      highest_num_clusters, highest_degree, kernel_type, Y_high_filter, min_count_per_day, 
              plot_graph = True, graph_type = 'regression', print_info = True)
```

Reference `example/example.py` for a demonstration and explanation on a small amount of provided data.

### Performance

Each color corresponds with a cluster, also known as a `type of day`.

![PVPolyfit regression](https://media.discordapp.net/attachments/210945856294223872/610932223952158723/unknown.png)

### Acknowledgements
![Emblems](https://media.discordapp.net/attachments/210945856294223872/610937089172963371/944bef9cdaef086d8659b5b825dd22c0.png)

