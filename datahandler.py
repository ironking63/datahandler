import pandas as pd
import numpy as np
import re


def _make_hashable(x):
    """Return a hashable stand-in for x.

    Pandas' duplicate-detection and counting operations (drop_duplicates,
    nunique, is_unique, value_counts) need to hash every value. Lists,
    dicts, sets, etc. aren't hashable and would otherwise crash these
    operations outright. Using each value's repr() as a stand-in lets
    equal unhashable values still compare as equal (two identical lists
    produce the same repr), and different values still compare as
    different, without altering any of the real data in the DataFrame.
    """
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


def _hashable_series(s):
    """Return a version of series `s` that is safe to hash (for nunique,
    is_unique, value_counts), substituting a repr-based proxy for any
    unhashable values in object-dtype columns. Non-object columns can't
    contain unhashable values, so they're returned unchanged."""
    if s.dtype == object:
        return s.map(_make_hashable)
    return s


@pd.api.extensions.register_dataframe_accessor("clean")
class DataFrameCleaner:
    """
    A lightweight pandas DataFrame cleaner accessor.

    Each method mutates the accessor's internal DataFrame and returns
    `self`, so calls can be chained directly off `.clean`. Because the
    chain returns the accessor (not a DataFrame), finish the chain with
    `.df` to get the actual cleaned DataFrame back.

    `lower()`, `upper()`, and `title()` are alternative capitalization
    steps — use at most one of them in a given chain.

    Usage:
        df = (
            df.clean
              .colnames()
              .dropempty()
              .dropdup()
              .na()
              .trim()
              .lower()
              .show()
              .dtnormal()
              .rmcats()
              .df
        )
    """

    def __init__(self, pandas_obj):
        self._df = pandas_obj

    @property
    def df(self):
        """Unwrap the accessor and return the underlying DataFrame."""
        return self._df

    # -----------------------------------
    # Clean column names
    # -----------------------------------
    def colnames(self):
        """Convert column names to snake_case."""
        df = self._df.copy()
        # .astype(str) (in addition to .map(str)) matters for an empty
        # DataFrame: mapping an empty Index never actually calls the
        # function, so pandas leaves its original dtype (e.g. int64 for
        # the default RangeIndex) instead of converting it, which would
        # otherwise crash the .str accessor calls below.
        df.columns = df.columns.map(str).astype(str).str.strip().str.lower()
        # A leading '-' (e.g. a column literally named -3) would otherwise
        # be stripped by the \W+ replacement below, silently turning -3
        # into "3" and losing the sign. Make it explicit first.
        df.columns = df.columns.str.replace(r"^-", "neg_", regex=True)
        df.columns = (
            df.columns.str.replace(r"\W+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        if df.columns.duplicated().any():
            seen = set()
            new_cols = []
            for col in df.columns:
                name = col
                i = 1
                while name in seen:
                    i += 1
                    name = f"{col}_{i}"
                seen.add(name)
                new_cols.append(name)
            df.columns = new_cols
        self._df = df
        return self

    # -----------------------------------
    # Remove empty rows/columns
    # -----------------------------------
    def dropempty(self):
        """Remove completely empty rows and columns."""
        self._df = self._df.dropna(how="all").dropna(axis=1, how="all")
        return self

    # -----------------------------------
    # Drop duplicate rows
    # -----------------------------------
    def dropdup(self):
        """Drop duplicate rows. Falls back to a hashable proxy comparison
        if any column contains unhashable values (e.g. lists/dicts), instead
        of crashing."""
        df = self._df
        try:
            self._df = df.drop_duplicates()
        except TypeError:
            proxy = df.apply(lambda col: _hashable_series(col))
            self._df = df[~proxy.duplicated()]
        return self

    # -----------------------------------
    # Replace blank strings with NaN
    # -----------------------------------
    def na(self):
        """Replace blank ('') strings with NaN in string columns."""
        df = self._df.copy()
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        df[obj_cols] = df[obj_cols].replace("", np.nan, regex=False)
        self._df = df
        return self

    # -----------------------------------
    # Trim leading/trailing spaces in all string cells
    # -----------------------------------
    def trim(self):
        """Trim whitespace in all string columns. Non-string values (numbers,
        booleans, lists, dicts, NaN/None) are left untouched, not nulled out."""
        df = self._df.copy()
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        self._df = df
        return self

    # -----------------------------------
    # Optional: lowercase all string columns
    # -----------------------------------
    def lower(self):
        """Convert all string columns to lowercase. Non-string values are left untouched."""
        df = self._df.copy()
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x)
        self._df = df
        return self

    # -----------------------------------
    # Optional: uppercase all string columns
    # -----------------------------------
    def upper(self):
        """Convert all string columns to uppercase. Non-string values are left untouched."""
        df = self._df.copy()
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].map(lambda x: x.upper() if isinstance(x, str) else x)
        self._df = df
        return self

    # -----------------------------------
    # Optional: title-case all string columns
    # -----------------------------------
    def title(self):
        """Convert all string columns to title case, e.g. 'john smith' -> 'John Smith'.
        Non-string values are left untouched."""
        df = self._df.copy()
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].map(lambda x: x.title() if isinstance(x, str) else x)
        self._df = df
        return self
        
    def outliers(self, cols=None, method='iqr', action='drop', threshold=1.5, z_threshold=3.0):
        """
        Detect and handle outliers in numeric columns.

        Parameters
        ----------
        cols : list of str, optional
            Columns to check. Defaults to all numeric columns.
        method : {'iqr', 'zscore'}
            Detection method. 'iqr' uses the interquartile range rule;
            'zscore' uses standard deviations from the mean.
        action : {'drop', 'replace'}
            What to do with detected outliers.
            'drop'    -- remove the entire row.
            'replace' -- replace the outlier value with NaN.
        threshold : float
            IQR multiplier (default 1.5). Only used when method='iqr'.
        z_threshold : float
            Z-score cutoff (default 3.0). Only used when method='zscore'.
    
        Returns
        -------
        self
        """
        import numpy as np
        df = self._df.copy()
        numeric_cols = cols if cols is not None else df.select_dtypes(include='number').columns.tolist()
    
        if method not in ('iqr', 'zscore'):
            raise ValueError("method must be 'iqr' or 'zscore'")
        if action not in ('drop', 'replace'):
            raise ValueError("action must be 'drop' or 'replace'")
    
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    
        for col in numeric_cols:
            series = df[col].dropna()
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
            elif method == 'zscore':
                mean = series.mean()
                std = series.std()
                if std == 0:
                    continue
                z_scores = (df[col] - mean) / std
                outlier_mask[col] = z_scores.abs() > z_threshold
    
        if action == 'drop':
            rows_to_drop = outlier_mask.any(axis=1)
            df = df[~rows_to_drop]
        elif action == 'replace':
            for col in numeric_cols:
                df.loc[outlier_mask[col], col] = np.nan
    
        self._df = df
        return self



    
    # -----------------------------------
    # Chain-friendly print preview
    # -----------------------------------
    def show(self, n=5):
        """Preview the top n rows without breaking the chain."""
        print(self._df.head(n))
        return self

    def dtnormal(self):
        df = self._df.copy()
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.normalize()
        self._df = df
        return self

    def rmcats(self):
        df = self._df.copy()
        for c in df.select_dtypes("category"):
            df[c] = df[c].cat.remove_unused_categories()
        self._df = df
        return self


def profile(df):
    records = []
    total = len(df)

    for i, col in enumerate(df.columns):
        # Index positionally (not by label) so that duplicate column names
        # don't cause df[col] to return a DataFrame instead of a Series.
        series = df.iloc[:, i]
        dtype = str(series.dtype)

        # --- Core stats ---
        value_count_ = series.count()
        value_count_percent = round(value_count_ / total * 100, 2) if total > 0 else 0.0
        safe_series = _hashable_series(series)
        distinct_count = safe_series.nunique(dropna=True)
        distinct_percent = round(distinct_count / value_count_ * 100, 2) if value_count_ > 0 else 0.0
        missing_count = series.isna().sum()
        missing_percent = round(missing_count / total * 100, 2) if total > 0 else 0.0
        is_unique_ = safe_series.is_unique

        # --- Type detection ---
        is_cat_dtype = isinstance(series.dtype, pd.CategoricalDtype)
        # A category column with numeric categories (e.g. a numeric-coded
        # categorical) should still get numeric stats, not fall through to
        # the generic '---' branch.
        numeric = pd.api.types.is_numeric_dtype(series) or (
            is_cat_dtype and pd.api.types.is_numeric_dtype(series.cat.categories)
        )
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        # --- Min / Max / Mean / Sum ---
        if numeric:
            # Categorical numeric columns need to be unwrapped to their
            # underlying numeric values before min/max/mean/sum will work.
            numeric_series = (
                pd.to_numeric(series.astype(object), errors="coerce") if is_cat_dtype else series
            )
            min_val = numeric_series.min()
            max_val = numeric_series.max()
            mean_val = round(numeric_series.mean(), 3) if total - missing_count > 0 else "---"
            sum_val = numeric_series.sum()
            empty_string = "---"
            zero_count = (numeric_series == 0).sum()
            negative_count = (numeric_series < 0).sum()
            positive_count = (numeric_series > 0).sum()
        elif is_datetime:
            non_na = series.dropna()
            min_val = non_na.min().date() if not non_na.empty else "---"
            max_val = non_na.max().date() if not non_na.empty else "---"
            mean_val = "---"
            sum_val = "---"
            empty_string = "---"
            zero_count = "---"
            negative_count = "---"
            positive_count = "---"
        else:
            min_val = "---"
            max_val = "---"
            mean_val = "---"
            sum_val = "---"
            empty_string = (series.isin([''])).sum()
            zero_count = "---"
            negative_count = "---"
            positive_count = "---"

        # ---------------------------
        # Prepare a frequency map aligned to display for datetimes
        # ---------------------------
        if is_datetime:
            # convert to date-only series for frequency calculations
            series_for_freq = series.dt.date
        else:
            series_for_freq = safe_series

        # ----------------------------------------
        # Top 10 by Count — Value - count (percent%)
        # (use series_for_freq so datetimes become dates here too)
        # ----------------------------------------
        top10_counts = series_for_freq.value_counts(dropna=True).head(10)
        top10_count_lines = []
        for value, count in top10_counts.items():
            # value is already date-only for datetimes
            percent = round((count / total) * 100, 2) if total > 0 else 0.0
            top10_count_lines.append(f"{value} - {count} ({percent}%)")
        top10_by_count = "\n".join(top10_count_lines)

        # ----------------------------------------
        # Assemble profile
        # ----------------------------------------
        records.append({
            "Type": dtype,
            "Uniqueness": is_unique_,
            "Total": f"{value_count_} ({value_count_percent}%)",
            "Missing": f"{missing_count} ({missing_percent}%)",
            "Unique": f"{distinct_count} ({distinct_percent}%)",
            "Empty string": empty_string,
            "Neg | Zero | Pos": f"{negative_count} | {zero_count} | {positive_count}",
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Sum": sum_val,
            "Top 10 Count": top10_by_count,
        })

    # --- Build the table, keeping every column's stats even if names repeat ---
    profile_df = pd.DataFrame(records, index=df.columns).T
    profile_df.index.name = "Rows: " + str(total)
    profile_df.columns.name = str(int(df.memory_usage(deep=True).sum() / 1000)) + " KB"

    return profile_df


def clean(df_):
    df = df_.copy()

    # 1. Drop completely empty rows and columns before touching column names
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Fix up column names: stringify, lowercase, preserve sign on
    # negative-number names, snake_case, then de-duplicate collisions.
    # .astype(str) (in addition to .map(str)) matters for an empty
    # DataFrame: mapping an empty Index never actually calls the function,
    # so pandas leaves its original dtype (e.g. int64 for the default
    # RangeIndex) instead of converting it, which would otherwise crash
    # the .str accessor calls below.
    df.columns = df.columns.map(str).astype(str).str.strip().str.lower()
    # A leading '-' (e.g. a column literally named -3) would otherwise be
    # stripped by the \W+ replacement below, silently turning -3 into "3"
    # and losing the sign. Make it explicit first.
    df.columns = df.columns.str.replace(r"^-", "neg_", regex=True)
    df.columns = df.columns.str.replace(r"\W+", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")

    if df.columns.duplicated().any():
        seen = set()
        new_cols = []
        for col in df.columns:
            name = col
            i = 1
            while name in seen:
                i += 1
                name = f"{col}_{i}"
            seen.add(name)
            new_cols.append(name)
        df.columns = new_cols

    # 4. Trim leading/trailing whitespace on all string columns first, so
    #    whitespace-only values (e.g. '   ') collapse to '' before the
    #    blank check below. Non-string values (numbers, booleans, lists,
    #    dicts, NaN/None) are left untouched, not nulled out.
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    # 2. Only replace blank ('') strings with NaN -- "na", "n/a", "none",
    #    and "missing" are left untouched as real string values.
    df[obj_cols] = df[obj_cols].replace("", np.nan, regex=False)

    try:
        df.drop_duplicates(inplace=True)
    except TypeError:
        proxy = df.apply(lambda col: _hashable_series(col))
        df = df[~proxy.duplicated()]

    return df


def rmcats(df):
    df = df.copy()
    for c in df.select_dtypes("category"):
        df[c] = df[c].cat.remove_unused_categories()
    return df
