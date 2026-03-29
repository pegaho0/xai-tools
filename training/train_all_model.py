from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_and_save_bundle(df, feature_cols, target_col, num_features, output_path):
    cat_features = [c for c in feature_cols if c not in num_features]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    clf = LogisticRegression(max_iter=4000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    feature_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    feature_names = [n.replace("cat__", "").replace("num__", "") for n in feature_names]

    X_bg = X.sample(n=min(300, len(X)), random_state=42)
    X_bg_trans = pipe.named_steps["pre"].transform(X_bg)

    explainer = shap.LinearExplainer(
        pipe.named_steps["clf"],
        X_bg_trans,
        feature_perturbation="interventional",
    )

    bundle = {
        "model": pipe,
        "explainer": explainer,
        "feature_names": feature_names,
        "num_features": num_features,
    }

    joblib.dump(bundle, output_path)
    print(f"Saved: {output_path}")


def create_pizza_catalog():
    rows = [
        ["MARGHERITA", "Margherita", "Italian", "Cheese", "Vegetarian", 18, 4.3, "Yes"],
        ["PEPPERONI", "Pepperoni", "American", "Pepperoni", "None", 22, 4.6, "No"],
        ["VEGGIE", "Veggie Supreme", "American", "Vegetables", "Vegetarian", 20, 4.4, "Yes"],
        ["CHICKEN_DELUXE", "Chicken Deluxe", "American", "Chicken", "None", 26, 4.7, "No"],
        ["MUSHROOM_TRUFFLE", "Mushroom Truffle", "Italian", "Mushrooms", "Vegetarian", 28, 4.8, "Yes"],
        ["GLUTEN_FREE_GARDEN", "Gluten-Free Garden", "Italian", "Vegetables", "Gluten-free", 31, 4.5, "No"],
        ["VEGAN_CLASSIC", "Vegan Classic", "Italian", "Vegetables", "Vegan", 24, 4.2, "Yes"],
        ["DAIRY_FREE_CHICKEN", "Dairy-Free Chicken", "American", "Chicken", "Dairy-free", 29, 4.4, "Yes"],
    ]
    df = pd.DataFrame(rows, columns=[
        "pizza_id", "name", "style", "ingredient", "dietary_tag", "price", "customer_rating", "free_delivery"
    ])
    df.to_csv(DATA_DIR / "pizza_catalog.csv", index=False)
    return df


def generate_pizza_training(n=3500, seed=7):
    rng = np.random.default_rng(seed)
    catalog = pd.read_csv(DATA_DIR / "pizza_catalog.csv")
    pizzas = catalog.to_dict(orient="records")

    def is_compatible(pizza_tag, user_restriction):
        if user_restriction in ["None", "Other"]:
            return True
        if user_restriction == "Vegetarian":
            return pizza_tag in ["Vegetarian", "Vegan"]
        if user_restriction == "Vegan":
            return pizza_tag == "Vegan"
        if user_restriction == "Gluten-free":
            return pizza_tag == "Gluten-free"
        if user_restriction == "Dairy-free":
            return pizza_tag in ["Dairy-free", "Vegan"]
        return True

    rating_weights = {
        "Not important": 0.0,
        "Slightly important": 0.7,
        "Moderately important": 1.4,
        "Very important": 2.1,
        "Extremely important": 2.8,
    }
    delivery_weights = {
        "Not important": 0.0,
        "Slightly important": 0.8,
        "Moderately important": 1.5,
        "Very important": 2.2,
        "Extremely important": 3.0,
    }

    max_price = rng.integers(15, 51, size=n)
    pizza_style = rng.choice(["Italian", "American"], size=n, p=[0.55, 0.45])
    ingredient_preference = rng.choice(["Cheese", "Pepperoni", "Chicken", "Vegetables", "Mushrooms"], size=n)
    dietary_restriction_model = rng.choice(["None", "Vegetarian", "Vegan", "Gluten-free", "Dairy-free", "Other"], size=n)
    rating_importance = rng.choice(list(rating_weights.keys()), size=n)
    free_delivery_importance = rng.choice(list(delivery_weights.keys()), size=n)

    labels = []
    for i in range(n):
        mp = int(max_price[i])
        stl = pizza_style[i]
        ing = ingredient_preference[i]
        dr = dietary_restriction_model[i]
        ri = rating_importance[i]
        fdi = free_delivery_importance[i]

        affordable = [p for p in pizzas if p["price"] <= mp]
        if not affordable:
            affordable = [min(pizzas, key=lambda x: x["price"])]

        compatible = [p for p in affordable if is_compatible(p["dietary_tag"], dr)]
        candidates = compatible if compatible else affordable

        def score(p):
            s = 0.0
            s += 2.2 if p["style"] == stl else 0.0
            s += 2.6 if p["ingredient"] == ing else 0.0
            s += 3.5 if is_compatible(p["dietary_tag"], dr) else -3.5
            s += rating_weights[ri] * (p["customer_rating"] - 4.0) * 2.2
            s += delivery_weights[fdi] if p["free_delivery"] == "Yes" else 0.0
            gap = max(0, p["price"] - mp)
            s -= 1.8 * gap
            s -= 0.05 * p["price"]
            s += rng.normal(0, 0.35)
            return s

        labels.append(max(candidates, key=score)["pizza_id"])

    return pd.DataFrame({
        "max_price": max_price,
        "pizza_style": pizza_style,
        "ingredient_preference": ingredient_preference,
        "dietary_restriction_model": dietary_restriction_model,
        "rating_importance": rating_importance,
        "free_delivery_importance": free_delivery_importance,
        "target": labels,
    })


def create_tour_catalog():
    rows = [
        ["TOUR_PARIS", "Paris Culture Escape", "Europe", "Mild", "Culture", "Couple", "Medium", 2600, 4.7],
        ["TOUR_SWISS", "Swiss Alps Adventure", "Europe", "Cold", "Adventure", "Friends", "Long", 4200, 4.8],
        ["TOUR_THAI", "Thailand Relax Retreat", "Asia", "Warm", "Relaxation", "Couple", "Medium", 2400, 4.6],
        ["TOUR_JAPAN", "Japan City and Food Tour", "Asia", "Mild", "Culture", "Solo", "Long", 3800, 4.9],
        ["TOUR_MOROCCO", "Morocco Heritage Journey", "Africa", "Warm", "Mixed", "Family", "Medium", 2200, 4.5],
        ["TOUR_CANADA", "Canadian Rockies Nature Tour", "North America", "Cold", "Nature", "Family", "Long", 3100, 4.7],
        ["TOUR_PERU", "Peru Adventure Explorer", "South America", "Mild", "Adventure", "Friends", "Long", 3300, 4.6],
        ["TOUR_DUBAI", "Dubai Luxury Short Break", "Middle East", "Warm", "Relaxation", "Couple", "Short", 2900, 4.4],
    ]
    df = pd.DataFrame(rows, columns=[
        "tour_id", "tour_name", "region", "climate", "travel_style", "group_type", "trip_duration", "price", "rating"
    ])
    df.to_csv(DATA_DIR / "tour_catalog.csv", index=False)
    return df


def generate_tour_training(n=5000, seed=11):
    rng = np.random.default_rng(seed)
    catalog = pd.read_csv(DATA_DIR / "tour_catalog.csv")
    tours = catalog.to_dict(orient="records")

    budget = rng.integers(800, 6001, size=n)
    trip_duration = rng.choice(["Short", "Medium", "Long"], size=n, p=[0.28, 0.42, 0.30])
    preferred_region = rng.choice(["Europe", "Asia", "North America", "South America", "Middle East", "Africa"], size=n)
    preferred_climate = rng.choice(["Cold", "Mild", "Warm"], size=n)
    travel_style = rng.choice(["Adventure", "Relaxation", "Culture", "Nature", "Mixed"], size=n)
    group_type = rng.choice(["Solo", "Couple", "Family", "Friends"], size=n)
    accommodation_level = rng.choice(["Budget", "Standard", "Premium", "Luxury"], size=n)
    food_interest = rng.choice(["Low", "Medium", "High"], size=n)
    transportation_comfort = rng.choice(["Basic", "Moderate", "High"], size=n)
    season = rng.choice(["Spring", "Summer", "Autumn", "Winter"], size=n)
    safety_importance = rng.choice(["Low", "Medium", "High", "Very high"], size=n)
    rating_importance = rng.choice(["Low", "Medium", "High", "Very high"], size=n)

    labels = []
    for i in range(n):
        user = {
            "budget": int(budget[i]),
            "trip_duration": trip_duration[i],
            "preferred_region": preferred_region[i],
            "preferred_climate": preferred_climate[i],
            "travel_style": travel_style[i],
            "group_type": group_type[i],
            "accommodation_level": accommodation_level[i],
            "food_interest": food_interest[i],
            "transportation_comfort": transportation_comfort[i],
            "season": season[i],
            "safety_importance": safety_importance[i],
            "rating_importance": rating_importance[i],
        }

        def score(t):
            s = 0.0
            s += 3.0 if t["region"] == user["preferred_region"] else 0.0
            s += 2.5 if t["climate"] == user["preferred_climate"] else 0.0
            s += 3.0 if t["travel_style"] == user["travel_style"] else 0.0
            s += 2.0 if t["group_type"] == user["group_type"] else 0.0
            s += 2.0 if t["trip_duration"] == user["trip_duration"] else 0.0
            gap = max(0, t["price"] - user["budget"])
            s -= gap / 250
            s += t["rating"] * (1.5 if user["rating_importance"] in ["High", "Very high"] else 0.8)
            if user["safety_importance"] in ["High", "Very high"] and t["region"] in ["Europe", "North America", "Asia"]:
                s += 1.2
            if user["food_interest"] == "High" and t["region"] in ["Asia", "Europe"]:
                s += 0.8
            s += rng.normal(0, 0.35)
            return s

        labels.append(max(tours, key=score)["tour_id"])

    return pd.DataFrame({
        "budget": budget,
        "trip_duration": trip_duration,
        "preferred_region": preferred_region,
        "preferred_climate": preferred_climate,
        "travel_style": travel_style,
        "group_type": group_type,
        "accommodation_level": accommodation_level,
        "food_interest": food_interest,
        "transportation_comfort": transportation_comfort,
        "season": season,
        "safety_importance": safety_importance,
        "rating_importance": rating_importance,
        "target": labels,
    })


def create_house_catalog():
    rows = [
        ["HOUSE_MTL_CONDO", "Downtown Montreal Condo", "Montreal", "Condo", 2, 2, 1100, 540000, "Yes", "No", "Good"],
        ["HOUSE_QC_FAMILY", "Quebec Family House", "Quebec City", "Detached house", 4, 3, 2400, 620000, "Yes", "Yes", "Good"],
        ["HOUSE_TORONTO_INVEST", "Toronto Urban Townhouse", "Toronto", "Townhouse", 3, 2, 1500, 880000, "Yes", "No", "Basic"],
        ["HOUSE_VAN_LUX", "Vancouver Scenic Villa", "Vancouver", "Detached house", 5, 4, 3200, 1450000, "Yes", "Yes", "Excellent"],
        ["HOUSE_CALGARY_VALUE", "Calgary Value Home", "Calgary", "Semi-detached", 3, 2, 1800, 460000, "Yes", "Yes", "Basic"],
        ["HOUSE_MTL_FAMILY", "Montreal Family Townhouse", "Montreal", "Townhouse", 3, 2, 1700, 590000, "Yes", "Yes", "Good"],
        ["HOUSE_QC_CONDO", "Quebec City River Condo", "Quebec City", "Condo", 2, 1, 950, 395000, "No", "No", "Excellent"],
        ["HOUSE_TORONTO_COMPACT", "Toronto Compact Condo", "Toronto", "Condo", 1, 1, 700, 510000, "No", "No", "Good"],
    ]
    df = pd.DataFrame(rows, columns=[
        "house_id", "listing_name", "city", "property_type", "bedrooms", "bathrooms", "area_size", "price", "parking", "garden", "view_quality"
    ])
    df.to_csv(DATA_DIR / "house_catalog.csv", index=False)
    return df


def generate_house_training(n=6000, seed=17):
    rng = np.random.default_rng(seed)
    catalog = pd.read_csv(DATA_DIR / "house_catalog.csv")
    houses = catalog.to_dict(orient="records")

    budget = rng.integers(150000, 1500001, size=n)
    city = rng.choice(["Montreal", "Quebec City", "Toronto", "Vancouver", "Calgary"], size=n)
    property_type = rng.choice(["Condo", "Townhouse", "Detached house", "Semi-detached"], size=n)
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n)
    bathrooms = rng.choice([1, 2, 3, 4], size=n)
    area_size = rng.integers(500, 4001, size=n)
    distance_to_downtown = rng.choice(["Very close", "Close", "Moderate", "Far"], size=n)
    public_transport_access = rng.choice(["Low", "Medium", "High"], size=n)
    school_quality = rng.choice(["Low", "Medium", "High"], size=n)
    safety = rng.choice(["Low", "Medium", "High", "Very high"], size=n)
    noise_level = rng.choice(["Low", "Medium", "High"], size=n)
    parking = rng.choice(["Yes", "No"], size=n)
    garden = rng.choice(["Yes", "No"], size=n)
    view_quality = rng.choice(["Basic", "Good", "Excellent"], size=n)
    building_age = rng.choice(["New", "Moderate", "Older"], size=n)
    investment_potential = rng.choice(["Low", "Medium", "High"], size=n)
    property_tax_sensitivity = rng.choice(["Low", "Medium", "High"], size=n)
    family_suitability = rng.choice(["Low", "Medium", "High"], size=n)

    labels = []
    for i in range(n):
        user = {
            "budget": int(budget[i]),
            "city": city[i],
            "property_type": property_type[i],
            "bedrooms": int(bedrooms[i]),
            "bathrooms": int(bathrooms[i]),
            "area_size": int(area_size[i]),
            "parking": parking[i],
            "garden": garden[i],
            "view_quality": view_quality[i],
            "family_suitability": family_suitability[i],
            "investment_potential": investment_potential[i],
        }

        def score(h):
            s = 0.0
            s += 3.0 if h["city"] == user["city"] else 0.0
            s += 2.5 if h["property_type"] == user["property_type"] else 0.0
            s += 1.4 * max(0, 3 - abs(h["bedrooms"] - user["bedrooms"]))
            s += 1.2 * max(0, 3 - abs(h["bathrooms"] - user["bathrooms"]))
            s += min(2.5, h["area_size"] / max(1, user["area_size"]))
            s += 1.8 if h["parking"] == user["parking"] else 0.0
            s += 1.8 if h["garden"] == user["garden"] else 0.0
            s += 2.0 if h["view_quality"] == user["view_quality"] else 0.0
            gap = max(0, h["price"] - user["budget"])
            s -= gap / 60000
            if user["family_suitability"] == "High" and h["bedrooms"] >= 3:
                s += 1.2
            if user["investment_potential"] == "High" and h["city"] in ["Toronto", "Vancouver", "Montreal"]:
                s += 1.0
            s += rng.normal(0, 0.35)
            return s

        labels.append(max(houses, key=score)["house_id"])

    return pd.DataFrame({
        "budget": budget,
        "city": city,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area_size": area_size,
        "distance_to_downtown": distance_to_downtown,
        "public_transport_access": public_transport_access,
        "school_quality": school_quality,
        "safety": safety,
        "noise_level": noise_level,
        "parking": parking,
        "garden": garden,
        "view_quality": view_quality,
        "building_age": building_age,
        "investment_potential": investment_potential,
        "property_tax_sensitivity": property_tax_sensitivity,
        "family_suitability": family_suitability,
        "target": labels,
    })


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    create_pizza_catalog()
    pizza_df = generate_pizza_training()
    train_and_save_bundle(
        pizza_df,
        feature_cols=[
            "max_price", "pizza_style", "ingredient_preference", "dietary_restriction_model",
            "rating_importance", "free_delivery_importance"
        ],
        target_col="target",
        num_features=["max_price"],
        output_path=MODEL_DIR / "pizza_bundle.joblib",
    )

    create_tour_catalog()
    tour_df = generate_tour_training()
    train_and_save_bundle(
        tour_df,
        feature_cols=[
            "budget", "trip_duration", "preferred_region", "preferred_climate", "travel_style",
            "group_type", "accommodation_level", "food_interest", "transportation_comfort",
            "season", "safety_importance", "rating_importance"
        ],
        target_col="target",
        num_features=["budget"],
        output_path=MODEL_DIR / "tour_bundle.joblib",
    )

    create_house_catalog()
    house_df = generate_house_training()
    train_and_save_bundle(
        house_df,
        feature_cols=[
            "budget", "city", "property_type", "bedrooms", "bathrooms", "area_size",
            "distance_to_downtown", "public_transport_access", "school_quality", "safety",
            "noise_level", "parking", "garden", "view_quality", "building_age",
            "investment_potential", "property_tax_sensitivity", "family_suitability"
        ],
        target_col="target",
        num_features=["budget", "bedrooms", "bathrooms", "area_size"],
        output_path=MODEL_DIR / "house_bundle.joblib",
    )

    print("All data files and model bundles were created successfully.")


if __name__ == "__main__":
    main()