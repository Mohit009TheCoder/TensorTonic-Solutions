import numpy as np

def impute_missing(X, strategy='mean'):
    # Convert input to numpy array
    X = np.array(X, dtype=object)

    # Replace "nan" string with np.nan
    X = np.where(X == "nan", np.nan, X).astype(float)

    # Handle 1D case
    if X.ndim == 1:
        if strategy == 'mean':
            stat = np.nanmean(X)
        else:
            stat = np.nanmedian(X)

        if np.isnan(stat):
            stat = 0.0

        X[np.isnan(X)] = stat
        return X

    # 2D case
    if strategy == 'mean':
        col_stat = np.nanmean(X, axis=0)
    else:
        col_stat = np.nanmedian(X, axis=0)

    # If entire column is NaN → replace with 0
    col_stat = np.nan_to_num(col_stat, nan=0.0)

    # Replace NaNs with column statistic
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_stat, inds[1])

    return X


# Test
X = [[1,"nan"],[3,5]]
print(impute_missing(X))

y = [["nan",2],["nan",4]]
print(impute_missing(y, strategy="median"))

z = [1,"nan",3,"nan",5]
print(impute_missing(z))