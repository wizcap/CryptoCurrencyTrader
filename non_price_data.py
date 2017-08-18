import time
from pytrends.request import TrendReq
from API_settings import google_username, google_password


def initialise_google_session():
    return TrendReq(google_username, google_password, custom_useragent='My Pytrends Script')


def google_trends_interest_over_time(pytrend_local, search_terms):
    pytrend_local.build_payload(kw_list=search_terms)

    interest_time_df = pytrend_local.interest_over_time()

    unix_times = convert_timestamp_to_unix_time(interest_time_df[search_terms[0]])

    return unix_times, interest_time_df[search_terms[0]].tolist()


def convert_timestamp_to_unix_time(timestamps):
    unix_times = []
    for i in range(len(timestamps.index)):
        unix_times.append(time.mktime(list(timestamps.index)[i].timetuple()))

    return unix_times
