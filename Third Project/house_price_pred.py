print("This is house price prediction model")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

house_data = pd.read_csv("F:/Machine Learning Hands On/Hands on ML/Third Project/House_Prices.csv")
print(house_data.head())

# Visualize relationships between features and price

# plt.scatter(house_data["size_sqft"], house_data["price_1000usd"], c='blue')
# plt.xlabel("Size (sqft)")
# plt.ylabel("Price ($)")
# plt.title("House Size vs Price")
# plt.show()


# plt.scatter(house_data["bedrooms"], house_data["price_1000usd"], c='blue')
# plt.xlabel("Bedrooms")
# plt.ylabel("Price ($)")
# plt.title("House Bedrooms vs Price")
# plt.show()


# plt.scatter(house_data["age_years"], house_data["price_1000usd"], c='blue')
# plt.xlabel("Age Years")
#plt.ylabel("Price ($)")
#plt.title("House Age or Year vs Price")
#plt.show()

#plt.scatter(house_data["distance_city_km"], house_data["price_1000usd"], c='blue')
#plt.xlabel("Distance (KM)")
#plt.ylabel("Price ($)")
#plt.title("House Distance vs Price")
#plt.show()



def rule_size_1(row):
    return 1 if row["size_sqft"] > 1500 else 0

    

rule_size_pred = house_data.apply(rule_size_1, axis=1)


rule_1_accuracy = (rule_size_pred == (house_data["price_1000usd"] > 300)).mean()
print("Rule 1 Size Accuracy:", rule_1_accuracy)


def rule_bedrooms_1(row):   
    return 1 if row["bedrooms"] >= 3 else 0

rule_bedrooms_pred = house_data.apply(rule_bedrooms_1, axis=1)

rule_2_accuracy = (rule_bedrooms_pred == (house_data["price_1000usd"] > 300)).mean()        
print("Rule 2 Bedrooms Accuracy:", rule_2_accuracy)


# def rule_age_1(row):
    # return 1 if row["age_years"] < 20 else 0

# rule_age_pred = house_data.apply(rule_age_1, axis=1)
# rule_3_accuracy = (rule_age_pred == (house_data["price_1000usd"] > 300)).mean()
# print("Rule 3 Age Accuracy:", rule_3_accuracy)

# def rule_distance_1(row):
    # return 1 if row["distance_city_km"] < 10 else 0

# rule_distance_pred = house_data.apply(rule_distance_1, axis=1)
# rule_4_accuracy = (rule_distance_pred == (house_data["price_1000usd"] > 300)).mean()
# print("Rule 4 Distance Accuracy:", rule_4_accuracy)


# def combined_rule(row):
    # if (row["size_sqft"] > 1500) and (row["bedrooms"] >= 3) and (row["age_years"] < 20) and (row["distance_city_km"] < 10):
        # return 1
    # else:
        # return 0
    
# combined_pred = house_data.apply(combined_rule, axis=1)
# combined_accuracy = (combined_pred == (house_data["price_1000usd"] > 300)).mean()
# print("Combined Rule Accuracy:", combined_accuracy)



# now categoriesd the price into 4 categories

# if a sqft is greater than 1500 and bedroom 3 or > 3 and < 20 years price is greater than 300 dollars

# def price_category(row):
    # if (row["size_sqft"] > 1500) and (row["bedrooms"] >= 3) and (row["age_years"] < 20) and (row["distance_city_km"] < 10):
        # return "High"
    # elif (row["size_sqft"] > 1000) and (row["bedrooms"] >= 2) and (row["age_years"] < 30):
        # return "Medium"
    # elif (row["size_sqft"] > 500) and (row["bedrooms"] >= 1):
        # return "Low"
    # else:
        # return "Very Low"
    

# house_data["price_category"] = house_data.apply(price_category, axis=1)
# print(house_data[["size_sqft", "bedrooms", "age_years", "distance_city_km", "price_1000usd", "price_category"]].head(10))


# instead of doing this messy rules checking match with your manual writted rules use direct correlation method 
# find best correlated features with price and use them for prediction in linwear regression models this statistic approach is useful and easy.

correlation_matrix = house_data.corr()
print(correlation_matrix["price_1000usd"].sort_values(ascending=False))





