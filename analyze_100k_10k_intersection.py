import json

import pandas as pd


def main():
    df = pd.read_csv("./data/bdd100k_10k_images_with_100k_metadata_intersection.csv")
    df = df[(df["same_split"]) & (df["image_split"] == "test") & (df["weather"] != "undefined") & (df["timeofday"] != "undefined")]
    print(df.describe())

    unique_weather_list: list[str] = df["weather"].unique().tolist()
    print("Unique values of 'weather':", unique_weather_list)

    unique_timeofday_list: list[str] = df["timeofday"].unique().tolist()
    print("Unique values of 'timeofday':", unique_timeofday_list)
    print()

    pair_map: map[str, int] = {}
    for weather in unique_weather_list:
        for timeofday in unique_timeofday_list:
            key = (weather, timeofday)
            pair_map[key] = 0

    print("How many times each pair occurs?")
    for key, _ in pair_map.items():
        count = ((df["weather"] == key[0]) & (df["timeofday"] == key[1])).sum().item()
        pair_map[key] = count
    pair_data = [{"weather": pair[0], "time_of_day": pair[1], "count": value} for pair, value in pair_map.items()]
    pair_data.sort(key=lambda item: item["count"], reverse=True)

    weather_time_data = pd.DataFrame(pair_data)
    weather_time_data.to_csv("./data/weather_timeofday_pair_data.csv")


if __name__ == "__main__":
    main()
