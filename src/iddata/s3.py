import datetime

import s3fs

_S3_PREFIX = "infectious-disease-data/data-raw/"


def get_versioned_file_path(glob_pattern: str, as_of: datetime.date) -> str:
    """
    Return the relative S3 path (after "data-raw/") of the most recent file matching glob_pattern that is dated on or
    before as_of.

    Parameters
    ----------
    glob_pattern : str
        Full S3 glob, e.g.
        "infectious-disease-data/data-raw/influenza-nhsn/nhsn-????-??-??.csv"
    as_of : datetime.date
        Reference date.

    Returns
    -------
    str
        Relative path after "data-raw/", e.g. "influenza-nhsn/nhsn-2024-10-05.csv"

    Raises
    ------
    FileNotFoundError
        If no file exists at or before as_of.
    """
    as_of_path = glob_pattern.replace("????-??-??", str(as_of))[len(_S3_PREFIX):]
    glob_results = s3fs.S3FileSystem(anon=True).glob(glob_pattern)
    all_file_paths = sorted([f[len(_S3_PREFIX):] for f in glob_results])
    eligible = [f for f in all_file_paths if f <= as_of_path]
    if not eligible:
        raise FileNotFoundError(f"No versioned file found on or before {as_of} matching {glob_pattern}")

    return eligible[-1]
