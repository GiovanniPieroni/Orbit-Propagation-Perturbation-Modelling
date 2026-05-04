from pathlib import Path
import sys
from os.path import abspath, dirname, relpath
from inspect import getfile, getframeinfo, getsource
from datetime import datetime, timedelta
import os
import requests
from sgp4.api import Satrec, days2mdhms
import numpy as np
from math import trunc


def prepare_directories():
    Path("../plots").mkdir(exist_ok=True)
    Path("../data").mkdir(exist_ok=True)


def getsourcefunc(func):
    path = abspath(getfile(func))
    caller = abspath(dirname(getframeinfo(sys._getframe(1)).filename))
    return [relpath(path, caller)], getsource(func)


def plot_filename(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'plots/{base_name}_{timestamp}.png'




def get_tle_data(identifier, filename='tle_cache.txt'):
    """
    Retrieves TLE data for a satellite by Name or NORAD ID.
    Checks local cache file first; if not found, fetches from CelesTrak.

    Args:
        identifier (str or int): Satellite Name (e.g., "GONETS-M 24") or NORAD ID (e.g., 54151).
        filename (str): Path to the local TLE cache file.

    Returns:
        list: A list of 3 strings [Name, Line1, Line2] or None if not found.
    """
    identifier = str(identifier).strip()

    # 1. Try to find TLE in the local file
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Iterate in groups of 3 (Name, Line 1, Line 2)
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break  # Incomplete record safety

            name_line = lines[i].strip()
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            # Check if identifier matches Name OR NORAD ID (in Line 1 or Line 2)
            # NORAD ID is at index 2-7 in TLE lines (e.g., ' 54151')
            norad_id_in_file = line1[2:7].strip()

            if identifier == name_line or identifier == norad_id_in_file:
                print(f"Found '{identifier}' in local cache.")
                return [name_line, line1, line2]

    # 2. If not found, fetch from CelesTrak
    print(f"'{identifier}' not in cache. Fetching from CelesTrak...")

    base_url = "https://celestrak.org/NORAD/elements/gp.php"

    # Determine if identifier is ID (digits) or Name
    params = {'FORMAT': 'TLE'}
    if identifier.isdigit():
        params['CATNR'] = identifier
    else:
        params['NAME'] = identifier

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise error for bad status codes

        data = response.text.strip().splitlines()

        if len(data) >= 3:
            # CelesTrak might return multiple results for a name (e.g., 'STARLINK').
            # We take the first valid 3-line block.
            tle_set = [data[0].strip(), data[1], data[2]]

            # 3. Append the new TLE to the cache file
            with open(filename, 'a') as f:
                # Add a newline if file exists and isn't empty
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    f.write('\n')
                f.write('\n'.join(tle_set))

            print(f"Successfully fetched and cached '{identifier}'.")
            return tle_set
        else:
            print(f"Error: CelesTrak returned incomplete data for '{identifier}'.")
            print("DEBUG: Raw CelesTrak response:")
            print(response.text)

            return None

    except requests.exceptions.RequestException as e:
        print(f"Network error fetching TLE: {e}")
        return None


def create_satrec(TLE_data):

    line1 = TLE_data[1]
    line2 = TLE_data[2]
    sat = Satrec.twoline2rv(line1, line2)
    return sat


def ragged_to_matrix(list_of_arrays, fill=np.nan):
    """
    Convert a list of 1D arrays of different lengths into a (n_runs x max_len) matrix
    using NaN padding.
    """
    max_len = max(len(arr) for arr in list_of_arrays)
    n = len(list_of_arrays)

    mat = np.full((n, max_len), fill)

    for k, arr in enumerate(list_of_arrays):
        L = len(arr)
        mat[k, :L] = arr

    return mat


def tle_days_to_date(sat, offset=0.0):
    # The offset is in MJD from the initial MDJ to the final MDJ
    JD = sat.jdsatepoch + sat.jdsatepochF + offset
    JD_0 = JD + 0.5
    L_1 = trunc(JD_0 + 68569)
    L_2 = trunc(4*L_1/146097)
    L_3 = L_1 - trunc((146097*L_2 + 3) / 4)
    L_4 = trunc(4000*(L_3+1)/1461001)
    L_5 = L_3 - trunc(1461*L_4/4) + 31
    L_6 = trunc(80*L_5/2447)
    L_7 = trunc(L_6/11)

    D = L_5 - trunc(2447*L_6/80)
    M = L_6 + 2 - 12*L_7
    Y = 100 * (L_2 - 49) + L_4 + L_7


    frac_day = JD_0 % 1
    H = int(frac_day * 24)
    m = int((frac_day * 24 - H) * 60)
    s = round(((frac_day * 24 - H) * 60 - m) * 60)

    date = [Y, M, D, H, m, s]


    year, days = get_new_epoch_from_offset(sat, offset)
    month, day, hour, minute, second = days2mdhms(year, days)
    second = round(second)
    if year < 57:
        year = year + 2000
    else:
        year = year + 1900


    err = np.array(date) - np.array([year, month, day, hour, minute, second])
    return date, err


def get_new_epoch_from_offset(sat, offset_days):
    """

    Argumentss:
        sat: satellite twoline2rv object
        offset_days: The offset in days (float).

    """

    # Years >= 57 are treated as 19xx, years < 57 are 20xx
    if sat.epochyr >= 57:
        full_year = 1900 + sat.epochyr
    else:
        full_year = 2000 + sat.epochyr


    start_of_current_year = datetime(full_year - 1, 12, 31)
    epoch_datetime = start_of_current_year + timedelta(days=sat.epochdays)

    # 3. Apply the offset
    new_datetime = epoch_datetime + timedelta(days=offset_days)

    # 4. Convert back to SGP4 format (Year, Days)

    # Get the new 2-digit year
    new_full_year = new_datetime.year
    new_year_sgp4 = new_full_year % 100

    # Calculate the new "days" value relative to the new year
    #  subtract "Dec 31 of the previous year"
    start_of_new_year = datetime(new_full_year - 1, 12, 31)

    #  use total_seconds() to preserve the fractional day precision
    difference = new_datetime - start_of_new_year
    new_days_sgp4 = difference.total_seconds() / 86400.0

    return new_year_sgp4, new_days_sgp4
