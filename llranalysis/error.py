import numpy as np


def bootstrap_error_mean(a, num_samples):
    return np.array(
        [
            np.mean(a[np.random.randint(0, len(a), size=(len(a)))])
            for i in range(num_samples)
        ]
    ).std(ddof=1)


def jackknife_error_mean(a):
    samples = np.array([np.mean(a[np.arange(len(a)) != i]) for i in range(len(a))])
    print("Bias :", samples.mean() - a.mean())
    return samples.std(ddof=1)


def calculate_error(a, num_samples=200, error_type="standard deviation"):
    if error_type == "bootstrap":
        return bootstrap_error_mean(a, num_samples)
    elif error_type == "jackknife":
        return jackknife_error_mean(a)
    elif error_type == "error on mean":
        return a.std(ddof=1) / np.sqrt(len(a))
    else:
        return a.std(ddof=1)


def calculate_error_set(a, num_samples=200, error_type="standard deviation"):
    return np.array(
        [calculate_error(a[:, i], num_samples, error_type) for i in range(a.shape[1])]
    )
