class ABcolumns:

    normal_columns = [
        'Distribution of A',
        'Distribution of B',
        'Shapiro P-Value of A',
        'Shapiro P-Value of B',
        'Shapiro W-Value of A',
        'Shapiro W-Value of B'
    ]

    non_normal_columns = [
        'Equal Variance',
        'Levene P-Value',
        'Levene F-Value',
        'T-Test P-Value',
        'T-Test T-Value'
    ]

    variance_columns = [
        'Equal Variance',
        'Levene P-Value',
        'Levene F-Value',
    ]

    ttest_columns = [
        'T-Test P-Value',
        'T-Test T-Value',
    ]

    mannw_columns = [
        'Mann Whitney U-Test P-Value',
        'Mann Whitney U-Test U-Value',
    ]

    ratio_columns = [
        'Mean Ratio',
        'Median Ratio'
    ]