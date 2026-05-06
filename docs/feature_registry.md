# Feature Registry

This registry documents the current feature set for Checkpoint 2 and separates offline training features from online serving features.

| Feature | Source | Offline | Online | Refresh | Leakage risk |
|---|---|---:|---:|---|---|
| room_segment | collector segment | True | True | weekly snapshot / user input | low |
| rooms_count | CIAN normalized listing | True | True | weekly snapshot | low |
| total_meters | CIAN normalized listing | True | True | per request / listing input | low |
| floor_ratio | floor / floors_count | True | True | per request / listing input | low |
| is_first_floor, is_last_floor | floor, floors_count | True | True | per request / listing input | low |
| district | CIAN listing | True | True | weekly snapshot / user input | low |
| underground | CIAN listing | True | True | weekly snapshot / user input | low |
| lat, lon | geocoder (Nominatim) of street + house_number + district | True | True | per request / listing input | low |
| geo_precision | geocoder result tier | True | True | per request / listing input | low |
| distance_to_center_km | haversine(lat, lon, Дворцовая 59.9386, 30.3141) | True | True | per request / listing input | low |
| distance_to_metro_km | haversine(lat, lon, named_station_lat, named_station_lon) | True | True | per request / listing input | low |
| metro_known | underground != 'unknown' and station resolved by geocoder | True | True | per request / listing input | low |
| district_rooms_median_price_per_sqm | market aggregate from cleaned CIAN snapshot | True | True | weekly snapshot | medium: compute on train split for experiments |
| district_rooms_ads_count | market aggregate from cleaned CIAN snapshot | True | True | weekly snapshot | low |
| underground_median_price_per_sqm | market aggregate from cleaned CIAN snapshot | True | True | weekly snapshot | medium: compute on train split for experiments |
| target_price_per_sqm, log_target_price_per_sqm | price / total_meters and its log1p | True | False | weekly snapshot | target, never use as input feature |
| price, log_price | CIAN listing | True | False | weekly snapshot | target, never use as input feature |
