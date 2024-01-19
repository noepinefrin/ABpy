import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from termcolor import colored
from scipy import stats
from .abcol import ABcolumns

class ABpy(ABcolumns):

    def __init__(self, data: pd.DataFrame, target_class: str, test_variables: list | tuple, significance_level: float = 0.05) -> None:

        self.unique_classes = data[target_class].unique()

        assert self.unique_classes.shape == (2, ), "You just compare only 2 targets at the same time."

        self.data_gb = data.groupby(target_class)
        self.group_a =  self.data_gb.get_group(self.unique_classes[0])
        self.group_b = self.data_gb.get_group(self.unique_classes[1])

        self.data = data
        self.target_class = target_class

        self.test_variables = test_variables
        self.significance_level = significance_level

        self.result_format = {
            'index': self.test_variables,
            'columns': self.normal_columns + self.variance_columns + self.ttest_columns + self.mannw_columns + self.ratio_columns
        }

    def apply(self, verbose: bool = True) -> pd.DataFrame:
        self.result = pd.DataFrame(**self.result_format)
        self.verbose = verbose

        for variable in self.test_variables:

            self._run(variable=variable)

            # If data is normally distributed
            if self._is_normally_distributed(variable=variable):

                # Before apply t-test looking at the variables has equal variance across the groups.
                if self._is_equal_variance(variable=variable):

                    # Independent Normal-distributed & equal variance t-test
                    self._t_test_significance(variable=variable, equal_var=True)

                else:

                    # Independent Normal-distributed & non-equal variance t-test
                    self._t_test_significance(variable=variable, equal_var=False)

            # If data is not normally distributed, then apply Mann Whitney U-Test
            else:

                # Is there a any statistically significance difference between medians
                self._mann_whitney_u_test(variable=variable)


        return self.result

    def _is_normally_distributed(self, variable: str) -> bool:
        group_a_w, group_a_pval = stats.shapiro(self.group_a[variable])
        group_b_w, group_b_pval = stats.shapiro(self.group_b[variable])

        a_pval_scientific = "{:.3e}".format(group_a_pval)
        b_pval_scientific = "{:.3e}".format(group_b_pval)

        group_a_significance: bool = group_a_pval > self.significance_level
        group_b_significance: bool = group_b_pval > self.significance_level
        group_a_dist = 'normal' if group_a_significance else 'non-normal'
        group_b_dist = 'normal' if group_b_significance else 'non-normal'

        values = [
            group_a_dist,
            group_b_dist,
            a_pval_scientific,
            b_pval_scientific,
            group_a_w,
            group_b_w
        ]

        for column, value in zip(self.normal_columns, values):
            self._update_result(variable=variable, column=column, value=value)

        if not (group_a_significance and group_b_significance):
            for column, value in zip(self.non_normal_columns, ['-']*5):
                self._update_result(variable=variable, column=column, value=value)
        else:
            for column, value in zip(self.mannw_columns, ['-']*2):
                self._update_result(variable=variable, column=column, value=value)

        if self.verbose:
            self.__dist_verb(variable, a_pval_scientific, b_pval_scientific, group_a_significance, group_b_significance)

        return group_a_significance and group_b_significance

    def _is_equal_variance(self, variable: str) -> bool:

        f_value, levene_p = stats.levene(self.group_a[variable], self.group_b[variable])

        pval_scientific = "{:.3e}".format(levene_p)

        is_equal_variance = levene_p > self.significance_level

        equal_variance = 'equal-variance' if is_equal_variance else 'not-equal-variance'

        values = [
            equal_variance,
            pval_scientific,
            f_value
        ]

        for column, value in zip(self.variance_columns, values):
            self._update_result(variable=variable, column=column, value=value)

        if self.verbose:
            self.__homogeneity_assumption_verb(variable, f_value, pval_scientific, is_equal_variance)

        return is_equal_variance

    def _t_test_significance(self, variable: str, equal_var: bool = False) -> bool:

        t_value, t_test_p = stats.ttest_ind(self.group_a[variable], self.group_b[variable], equal_var=equal_var)

        pval_scientific = "{:.3e}".format(t_test_p)

        values = [
            pval_scientific,
            t_value
        ]

        for column, value in zip(self.ttest_columns, values):
            self._update_result(variable=variable, column=column, value=value)

        self._mean(variable=variable)

        is_significant = t_test_p > self.significance_level

        if self.verbose:
            self.__ttest_verb(variable, t_value, pval_scientific, is_significant)

        return is_significant

    def _mann_whitney_u_test(self, variable: str) -> bool:
        u_value, mannw_test_pval = stats.mannwhitneyu(self.group_a[variable], self.group_b[variable])

        pval_scientific = "{:.3e}".format(mannw_test_pval)

        values = [
            pval_scientific,
            u_value
        ]

        for column, value in zip(self.mannw_columns, values):
            self._update_result(variable=variable, column=column, value=value)

        self._median(variable=variable)

        is_significant = mannw_test_pval > self.significance_level

        if self.verbose:
            pass

        return is_significant

    def _mean(self, variable: str) -> None:
        group_a_mean = self.group_a.loc[:, variable].mean()
        group_b_mean = self.group_b.loc[:, variable].mean()

        is_greater = group_a_mean > group_b_mean
        mean_ratio = group_a_mean / group_b_mean

        values = [mean_ratio, '-']

        for column, value in zip(self.ratio_columns, values):
            self._update_result(variable=variable, column=column, value=value)

        if self.verbose:
            if is_greater:
                print(f'\nMean of A in {variable} is greater than B', '\n')
            else:
                print(f'\nMean of B in {variable} is greater than A', '\n')


    def _median(self, variable: str) -> None:
        group_a_median = self.group_a.loc[:, variable].median()
        group_b_median = self.group_b.loc[:, variable].median()

        is_greater = group_a_median > group_b_median
        median_ratio = group_a_median / group_b_median

        values = ['-', median_ratio]

        for column, value in zip(self.ratio_columns, values):
            self._update_result(variable=variable, column=column, value=value)

        if self.verbose:
            if is_greater:
                print(f'\nMedian of A in {variable} is greater than B', '\n')
            else:
                print(f'\nMedian of B in {variable} is greater than A', '\n')

    def _update_result(self, variable: str, column: str, value: int | float | str) -> None:
        self.result.loc[variable, column] = value

    def _run(self, variable):
        if self.verbose:
            print(
            colored(f'-'*85, 'magenta', attrs=['bold']),
            '\n',
            colored(f' A/B Testing for {variable} ', 'white', 'on_magenta', attrs=['bold', 'reverse', 'blink']),
            '\n',
            '\n',
            colored(f' Summary Statistics by Groups for {variable} ', 'white', 'on_blue', attrs=['bold', 'reverse', 'blink']),
            '\n', sep=''
            )

            desc_agg = self.data.groupby(self.target_class)[variable].aggregate(['count', 'mean', 'std', 'median', 'min', 'max']).T

            print(colored(desc_agg, 'blue', attrs=['bold']), '\n')

            plt.style.use('fivethirtyeight')

            self._draw_hist(variable=variable)
            self._draw_boxplot(variable=variable)

    def _draw_hist(self, variable) -> None:
        print(colored(f' Histogram by groups for {variable} ', 'white', 'on_light_green', attrs=['bold', 'reverse', 'blink']))

        sns.displot(
            data=self.data,
            x=variable,
            palette='crest',
            hue=self.target_class,
            kde=True)

        plt.title(f'{variable}\'s dist plot w/ kde')

        plt.show()

    def _draw_boxplot(self, variable) -> None:
        print(colored(f' Boxplot by groups for {variable} ', 'white', 'on_light_red', attrs=['bold', 'reverse', 'blink']))

        ax = sns.violinplot(
            x=self.target_class,
            y=variable,
            hue=self.target_class,
            data=self.data,
            palette='viridis',
            dodge=False,
            density_norm='width',
            inner=None)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for violin in ax.collections:
            bbox = violin.get_paths()[0].get_extents()
            x0, y0, width, height = bbox.bounds
            violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

        sns.boxplot(
            x=self.target_class,
            y=variable,
            data=self.data,
            saturation=1,
            showfliers=False,
            width=0.3,
            boxprops={'zorder': 3, 'facecolor': 'none'},
            ax=ax)

        old_len_collections = len(ax.collections)

        sns.stripplot(
            x=self.target_class,
            y=variable,
            data=self.data,
            dodge=False,
            color='#277f8e',
            s=10,
            marker="d",
            linewidth=1,
            alpha=.45,
            ax=ax)

        plt.title(f'{variable}\'s splitted Violin & Box & Strip Plot')

        for dots in ax.collections[old_len_collections:]:
            dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set(xlabel='Classes', ylabel=variable)
        plt.show()

    def __dist_verb(self, variable: str, a_pval: str, b_pval: str, a_significance: bool, b_significance: bool) -> None:
        print(
            colored(f' 1. Step: Testing the Normality Assumption for {variable} using Shaphiro Wilk Test', 'white', 'on_cyan', attrs=['bold', 'reverse', 'blink']),
            '\n',
            '\n',
            f'A P-Value: {a_pval}\nB P-Value: {b_pval}',
            '\n',
            sep=''
        )

        if a_significance and b_significance:
            print(
                f'Shaphiro Wilk Test resulted as p > {self.significance_level} for A and B which indicates that H0 can NOT be rejected. ',
                '\n',
                f'Accordingly distribution of {variable} values in A and B are likely to normal distribution.',
                '\n',
                sep=''
            )

        elif not (a_significance and b_significance):
            print(
                f'Shaphiro Wilk Test resulted as p < {self.significance_level} for A and B which indicates that H0 is rejected. ',
                '\n',
                f'Accordingly distribution of {variable} values in A and B are NOT likely to normal distribution.',
                '\n',
                sep=''
            )

        elif a_significance and not b_significance:
            print(
                colored(f' PAY ATTENTION ', 'white', 'on_red', attrs=['bold', 'reverse', 'blink']),
                '\t',
                f'Shaphiro Wil Test resulted as p > {self.significance_level} for A while p < {self.significance_level} for B which indicates that H0 is rejected for B.',
                f'Accordingly you can check if the {variable} values in B may contain outlier.',
                '\n',
                sep=''
            )

        elif not a_significance and b_significance: # also we can use else statement but this is more clear.
            print(
                colored(f' PAY ATTENTION ', 'white', 'on_red', attrs=['bold', 'reverse', 'blink']),
                '\t',
                f'Shaphiro Wil Test resulted as p > {self.significance_level} for B while p < {self.significance_level} for A which indicates that H0 is rejected for A.',
                f'Accordingly you can check if the {variable} values in A may contain outlier.',
                '\n',
                sep=''
            )

    def __homogeneity_assumption_verb(self, variable: str, levene_f: str, levene_pval: str, is_equal_variance: bool) -> None:
        print(
            colored(f' 2. Step: Testing the Homogeneity Assumption for {variable} using Levene\'s F-Test', 'white', 'on_cyan', attrs=['bold', 'reverse', 'blink']),
            '\n',
            '\n',
            f'Levene P-Value: {levene_pval} & B Levene F-Value: {levene_f}',
            '\n',
            sep=''
        )

        if is_equal_variance:
            print(
                f'Levene\'s F-Test resulted as p > {self.significance_level} for A and B which indicates that H0 can NOT be rejected. ',
                '\n',
                f'Accordingly variance of {variable} values in A and B are equal. ',
                sep=''
            )

    def __ttest_verb(self, variable, ttest_t, ttest_p, is_significant):
        pass