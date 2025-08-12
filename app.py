import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
import joblib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for datasets and encoders
refrig_df = None
refrig_le_company = None
refrig_le_type = None
refrig_le_problem = None
refrig_clf = None
refrig_reg = None

wm_df = None
wm_label_encoders = {}
wm_repairability_model = None
wm_cost_model = None

printer_df = None
printer_le_brand = None
printer_le_type = None
printer_le_problem = None
printer_clf = None
printer_reg = None

appliance_df = None
appliance_le_appliance = None
appliance_le_subtype = None
appliance_le_problem = None
appliance_clf = None
appliance_reg = None

mobile_df = None
mobile_label_encoders = {}
mobile_clf = None
mobile_reg = None

battery_df = None
df_original = None
label_encoders = {}

# --- Refrigerator Prediction Logic ---
try:
    refrig_df = pd.read_csv('refrigerator_repair_dataset.csv')
    refrig_le_company = LabelEncoder()
    refrig_le_type = LabelEncoder()
    refrig_le_problem = LabelEncoder()
    refrig_df['Company'] = refrig_le_company.fit_transform(refrig_df['Company'])
    refrig_df['Type'] = refrig_le_type.fit_transform(refrig_df['Type'])
    refrig_df['Problem'] = refrig_le_problem.fit_transform(refrig_df['Problem'])
    refrig_X_clf = refrig_df[['Company', 'Type', 'Year_of_Purchase', 'Problem']]
    refrig_y_clf = refrig_df['Decision'].map({'REPAIR': 1, 'REPLACE': 0})
    refrig_df_repair = refrig_df[refrig_df['Decision'] == 'REPAIR']
    refrig_X_reg = refrig_df_repair[['Company', 'Type', 'Year_of_Purchase', 'Problem']]
    refrig_y_reg = refrig_df_repair['Cost']
    refrig_clf = RandomForestClassifier(random_state=42)
    refrig_clf.fit(refrig_X_clf, refrig_y_clf)
    refrig_reg = RandomForestRegressor(random_state=42)
    refrig_reg.fit(refrig_X_reg, refrig_y_reg)
except FileNotFoundError:
    logger.error("refrigerator_repair_dataset.csv not found.")

def predict_refrigerator(company, type_, year, problem):
    if refrig_df is None:
        return "Dataset not loaded"
    try:
        company_enc = refrig_le_company.transform([company])[0]
        type_enc = refrig_le_type.transform([type_])[0]
        problem_enc = refrig_le_problem.transform([problem])[0]
        input_data = pd.DataFrame({
            'Company': [company_enc],
            'Type': [type_enc],
            'Year_of_Purchase': [year],
            'Problem': [problem_enc]
        })
        clf_pred = refrig_clf.predict(input_data)[0]
        if clf_pred == 0:
            return "Replace"
        else:
            cost_pred = refrig_reg.predict(input_data)[0]
            return f"Repair, Estimated Cost: ₹{cost_pred:.2f}"
    except Exception as e:
        logger.error(f"Error in refrigerator prediction: {str(e)}")
        return "Invalid input values"

# --- Washing Machine Prediction Logic ---
try:
    wm_df = pd.read_csv('wm.csv')
    wm_df['Cost'] = wm_df['Cost'].replace(-1, np.nan)
    median_cost = wm_df['Cost'].median()
    wm_df['Cost'] = wm_df['Cost'].fillna(median_cost)
    wm_df['Repairable'] = wm_df['Repair_or_Replace'].apply(lambda x: 1 if x == 'Repairable' else 0)
    wm_categorical_cols = ['Company', 'Type', 'Problem']
    for col in wm_categorical_cols:
        le = LabelEncoder()
        wm_df[col] = le.fit_transform(wm_df[col])
        wm_label_encoders[col] = le
    wm_X = wm_df[['Company', 'Type', 'Year_of_Purchase', 'Problem']]
    wm_y_repairable = wm_df['Repairable']
    wm_repairable_df = wm_df[wm_df['Repairable'] == 1]
    wm_X_cost = wm_repairable_df[['Company', 'Type', 'Year_of_Purchase', 'Problem']]
    wm_y_cost = wm_repairable_df['Cost']
    wm_repairability_model = RandomForestClassifier(n_estimators=100, random_state=42)
    wm_repairability_model.fit(wm_X, wm_y_repairable)
    wm_cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
    wm_cost_model.fit(wm_X_cost, wm_y_cost)
except FileNotFoundError:
    logger.error("wm.csv not found.")

def predict_washing_machine(company, type_, year, problem):
    if wm_df is None:
        return "Dataset not loaded"
    try:
        logger.debug(f"Input: company={company}, type={type_}, year={year}, problem={problem}")
        company_enc = wm_label_encoders['Company'].transform([company])[0]
        type_enc = wm_label_encoders['Type'].transform([type_])[0]
        problem_enc = wm_label_encoders['Problem'].transform([problem])[0]
        input_data = pd.DataFrame({
            'Company': [company_enc],
            'Type': [type_enc],
            'Year_of_Purchase': [int(year)],
            'Problem': [problem_enc]
        })
        repairable_prob = wm_repairability_model.predict_proba(input_data)[0][1]
        is_repairable = repairable_prob > 0.5
        if is_repairable:
            predicted_cost = wm_cost_model.predict(input_data)[0]
            return f"Repair, Estimated Cost: ₹{predicted_cost:.2f}"
        else:
            return "Replace"
    except Exception as e:
        logger.error(f"Error in washing machine prediction: {str(e)}")
        return "Invalid input values"

# --- Printer Prediction Logic ---
try:
    printer_df = pd.read_csv('printer_xerox_repair_dataset.csv')
    printer_le_brand = LabelEncoder()
    printer_le_type = LabelEncoder()
    printer_le_problem = LabelEncoder()
    printer_df['Brand'] = printer_le_brand.fit_transform(printer_df['Brand'])
    printer_df['Type'] = printer_le_type.fit_transform(printer_df['Type'])
    printer_df['Problem'] = printer_le_problem.fit_transform(printer_df['Problem'])
    printer_X_clf = printer_df[['Brand', 'Type', 'Year_of_Purchase', 'Problem']]
    printer_y_clf = printer_df['Decision'].map({'REPAIR': 1, 'REPLACE': 0})
    printer_df_repair = printer_df[printer_df['Decision'] == 'REPAIR']
    printer_X_reg = printer_df_repair[['Brand', 'Type', 'Year_of_Purchase', 'Problem']]
    printer_y_reg = printer_df_repair['Cost']
    printer_clf = RandomForestClassifier(random_state=42)
    printer_clf.fit(printer_X_clf, printer_y_clf)
    printer_reg = RandomForestRegressor(random_state=42)
    printer_reg.fit(printer_X_reg, printer_y_reg)
except FileNotFoundError:
    logger.error("printer_xerox_repair_dataset.csv not found.")

def predict_printer(brand, type_, year, problem):
    if printer_df is None:
        return "Dataset not loaded"
    try:
        brand_enc = printer_le_brand.transform([brand])[0]
        type_enc = printer_le_type.transform([type_])[0]
        problem_enc = printer_le_problem.transform([problem])[0]
        input_data = pd.DataFrame({
            'Brand': [brand_enc],
            'Type': [type_enc],
            'Year_of_Purchase': [year],
            'Problem': [problem_enc]
        })
        clf_pred = printer_clf.predict(input_data)[0]
        if clf_pred == 0:
            return "Replace"
        else:
            cost_pred = printer_reg.predict(input_data)[0]
            return f"Repair, Estimated Cost: ₹{cost_pred:.2f}"
    except Exception as e:
        logger.error(f"Error in printer prediction: {str(e)}")
        return "Invalid input values"

# --- Appliance Prediction Logic ---
try:
    appliance_df = pd.read_csv('appliance_diagnosis_dataset.csv')
    appliance_df['RepairCost'] = pd.to_numeric(appliance_df['RepairCost'], errors='coerce')
    median_cost = appliance_df['RepairCost'].median()
    appliance_df['RepairCost'] = appliance_df['RepairCost'].fillna(median_cost)
    appliance_le_appliance = LabelEncoder()
    appliance_le_subtype = LabelEncoder()
    appliance_le_problem = LabelEncoder()
    appliance_df['ApplianceType'] = appliance_le_appliance.fit_transform(appliance_df['ApplianceType'])
    appliance_df['SubType'] = appliance_le_subtype.fit_transform(appliance_df['SubType'])
    appliance_df['Problem'] = appliance_le_problem.fit_transform(appliance_df['Problem'])
    # Updated to current date: August 09, 2025
    appliance_df['YearDiff'] = 2025 - appliance_df['Year']
    appliance_X_clf = appliance_df[['ApplianceType', 'SubType', 'Problem', 'YearDiff']]
    appliance_y_clf = appliance_df['Repairability'].map({'Repairable': 1, 'Replaceable': 0})
    appliance_df_repair = appliance_df[appliance_df['Repairability'] == 'Repairable']
    appliance_X_reg = appliance_df_repair[['ApplianceType', 'SubType', 'Problem', 'YearDiff']]
    appliance_y_reg = appliance_df_repair['RepairCost']
    appliance_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    appliance_clf.fit(appliance_X_clf, appliance_y_clf)
    appliance_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    if not appliance_X_reg.empty and not appliance_y_reg.empty:
        appliance_reg.fit(appliance_X_reg, appliance_y_reg)
except FileNotFoundError:
    logger.error("appliance_diagnosis_dataset.csv not found.")

def predict_appliance(appliance_type, subtype, year, problem):
    if appliance_df is None:
        return "Dataset not loaded"
    try:
        logger.debug(f"Input: appliance_type={appliance_type}, subtype={subtype}, year={year}, problem={problem}")
        if appliance_type not in appliance_le_appliance.classes_:
            return f"Invalid appliance type: {appliance_type} not recognized"
        if subtype not in appliance_le_subtype.classes_:
            return f"Invalid subtype: {subtype} not recognized"
        if problem not in appliance_le_problem.classes_:
            return f"Invalid problem: {problem} not recognized"
        appliance_enc = appliance_le_appliance.transform([appliance_type])[0]
        subtype_enc = appliance_le_subtype.transform([subtype])[0]
        problem_enc = appliance_le_problem.transform([problem])[0]
        year_diff = 2025 - int(year)  # Updated to 2025
        input_data = pd.DataFrame({
            'ApplianceType': [appliance_enc],
            'SubType': [subtype_enc],
            'Problem': [problem_enc],
            'YearDiff': [year_diff]
        })
        clf_pred = appliance_clf.predict(input_data)[0]
        if clf_pred == 0:
            return "Replace"
        else:
            cost_pred = appliance_reg.predict(input_data)[0]
            return f"Repair, Estimated Cost: ₹{cost_pred:.2f}"
    except Exception as e:
        logger.error(f"Error in appliance prediction: {str(e)}")
        return "Invalid input values"

# --- Mobile Phone Prediction Logic ---
try:
    mobile_df = pd.read_csv('mobile_phone_repair_dataset.csv')
    mobile_df['Cost'] = mobile_df['Cost'].replace(-1, np.nan)
    median_cost = mobile_df['Cost'].median()
    mobile_df['Cost'] = mobile_df['Cost'].fillna(median_cost)
    mobile_df['Repairable'] = mobile_df['Decision'].apply(lambda x: 1 if x == 'REPAIR' else 0)
    mobile_categorical_cols = ['Brand', 'Feature', 'OS', 'Problem']
    for col in mobile_categorical_cols:
        le = LabelEncoder()
        mobile_df[col] = le.fit_transform(mobile_df[col])
        mobile_label_encoders[col] = le
    mobile_X_clf = mobile_df[['Brand', 'Feature', 'OS', 'Year_of_Purchase', 'Problem']]
    mobile_y_clf = mobile_df['Repairable']
    mobile_df_repair = mobile_df[mobile_df['Repairable'] == 1]
    mobile_X_reg = mobile_df_repair[['Brand', 'Feature', 'OS', 'Year_of_Purchase', 'Problem']]
    mobile_y_reg = mobile_df_repair['Cost']
    mobile_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    mobile_clf.fit(mobile_X_clf, mobile_y_clf)
    mobile_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    if not mobile_X_reg.empty and not mobile_y_reg.empty:
        mobile_reg.fit(mobile_X_reg, mobile_y_reg)
except FileNotFoundError:
    logger.error("mobile_phone_repair_dataset.csv not found.")

def predict_mobile_phone(brand, feature, os, year, problem):
    if mobile_df is None:
        return "Dataset not loaded"
    try:
        logger.debug(f"Input: brand={brand}, feature={feature}, os={os}, year={year}, problem={problem}")
        brand_enc = mobile_label_encoders['Brand'].transform([brand])[0]
        feature_enc = mobile_label_encoders['Feature'].transform([feature])[0]
        os_enc = mobile_label_encoders['OS'].transform([os])[0]
        problem_enc = mobile_label_encoders['Problem'].transform([problem])[0]
        input_data = pd.DataFrame({
            'Brand': [brand_enc],
            'Feature': [feature_enc],
            'OS': [os_enc],
            'Year_of_Purchase': [int(year)],
            'Problem': [problem_enc]
        })
        clf_pred = mobile_clf.predict(input_data)[0]
        if clf_pred == 0:
            return "Replace"
        else:
            cost_pred = mobile_reg.predict(input_data)[0]
            return f"Repair, Estimated Cost: ₹{cost_pred:.2f}"
    except Exception as e:
        logger.error(f"Error in mobile phone prediction: {str(e)}")
        return "Invalid input values"

# --- Battery Prediction Logic ---
try:
    battery_df = pd.read_csv('battery_problems_1000.csv')
except FileNotFoundError:
    logger.error("battery_problems_1000.csv not found.")

def predict_battery(battery_type, brand, problem):
    if battery_df is None:
        return 'No', 'Dataset not loaded', -1.0
    match = battery_df[(battery_df['Battery_Type'] == battery_type) & (battery_df['Brand'] == brand) & (battery_df['Problem'] == problem)]
    if not match.empty:
        repairable = match.iloc[0]['Repairable']
        action = match.iloc[0]['Action']
        cost = -1.0
        if repairable == 'Yes':
            parts = action.split('₹')
            if len(parts) > 1:
                cost_str = parts[-1].strip()
                try:
                    cost = float(cost_str)
                except ValueError:
                    cost = -1.0
        return repairable, action, cost
    else:
        return 'No', 'Not Repairable - Need to Buy New One', -1.0

# --- Resell Prediction Logic ---
try:
    df_original = pd.read_csv("resell_data.csv")
    # Normalize data: strip spaces, convert to title case, handle special characters
    for col in ["Appliance", "Brand", "Problem"]:
        df_original[col] = df_original[col].astype(str).str.strip().str.title()
    # Ensure Resell_Cost is numeric
    df_original["Resell_Cost"] = pd.to_numeric(df_original["Resell_Cost"], errors='coerce')
    df_original = df_original.dropna(subset=["Resell_Cost"])  # Drop rows with invalid Resell_Cost
    # Encode categorical variables for ML model
    df_encoded = df_original.copy()
    for col in ["Brand", "Appliance", "Problem"]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    # Train ML model
    X = df_encoded[["Brand", "Appliance", "Problem", "Age"]]
    y = df_original["Resell_Cost"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    # Save model and encoders
    joblib.dump(model, "model.pkl")
    joblib.dump(label_encoders, "encoders.pkl")
    logger.info("Resell model and encoders loaded successfully")
except FileNotFoundError:
    logger.error("resell_data.csv not found.")
except Exception as e:
    logger.error(f"Error loading resell_data.csv: {str(e)}")

# Define subtype and problem categories
subtype_categories = {
    'TV': ['Old TV', 'LED', 'LCD'],
    'Computer': ['Laptop', 'Desktop']
}

problem_categories = {
    'TV': {
        'Old TV': ['Color distortion', 'HDMI not working', 'Black screen but power light on', 
                   'Burning smell', 'No sound', 'Panel cracked', 'Tube damage', 'No signal detection'],
        'LED': ['HDMI not working', 'Color distortion', 'Black screen but power light on', 
                'Panel cracked', 'Burning smell', 'No sound', 'No signal detection'],
        'LCD': ['Color distortion', 'Black screen but power light on', 'Panel cracked', 
                'Burning smell', 'No sound', 'No signal detection']
    },
    'Computer': {
        'Laptop': ['HDD failure', 'Overheating', 'Battery not charging', 'Slow booting', 
                   'No power at all', 'Fan noise', 'Screen flickering', 'Motherboard dead'],
        'Desktop': ['HDD failure', 'Overheating', 'Slow booting', 'No power at all', 
                    'Fan noise', 'Screen flickering', 'Motherboard dead']
    }
}

# Flask Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/gallery', methods=['GET'])
def gallery():
    return render_template('gallery.html')

@app.route('/sell', methods=['GET'])
def sell():
    appliances = sorted(df_original["Appliance"].unique().tolist()) if df_original is not None else []
    logger.debug(f"Appliances available: {appliances}")
    return render_template('sell.html', appliances=appliances)

@app.route('/get_problems', methods=['POST'])
def get_problems():
    appliance = request.json.get("appliance", "").strip().title()
    logger.debug(f"Received appliance in /get_problems: {appliance}")
    if df_original is None:
        logger.error("Dataset not loaded for /get_problems")
        return jsonify({"problems": [], "brands": [], "error": "Dataset not loaded"}), 400
    filtered_df = df_original[df_original["Appliance"] == appliance]
    problems = sorted(filtered_df["Problem"].unique().tolist())
    brands = sorted(filtered_df["Brand"].unique().tolist())
    logger.debug(f"Problems for {appliance}: {problems}")
    logger.debug(f"Brands for {appliance}: {brands}")
    if not problems or not brands:
        logger.warning(f"No problems or brands found for appliance: {appliance}")
    return jsonify({"problems": problems, "brands": brands})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        brand = request.form.get("brand", "").strip().title()
        appliance = request.form.get("appliance", "").strip().title()
        problem = request.form.get("problem", "").strip().title()
        age = request.form.get("age")
        logger.debug(f"Predict input: brand={brand}, appliance={appliance}, problem={problem}, age={age}")
        age = int(age) if age else 0
        model = joblib.load("model.pkl")
        encoders = joblib.load("encoders.pkl")
        if brand not in encoders["Brand"].classes_:
            logger.error(f"Invalid brand: {brand}")
            return jsonify({"error": f"Invalid brand: {brand} not recognized"}), 400
        if appliance not in encoders["Appliance"].classes_:
            logger.error(f"Invalid appliance: {appliance}")
            return jsonify({"error": f"Invalid appliance: {appliance} not recognized"}), 400
        if problem not in encoders["Problem"].classes_:
            logger.error(f"Invalid problem: {problem}")
            return jsonify({"error": f"Invalid problem: {problem} not recognized"}), 400
        brand_enc = encoders["Brand"].transform([brand])[0]
        appliance_enc = encoders["Appliance"].transform([appliance])[0]
        problem_enc = encoders["Problem"].transform([problem])[0]
        pred_price = model.predict([[brand_enc, appliance_enc, problem_enc, age]])[0]
        logger.debug(f"Predicted resell price: {pred_price}")
        return jsonify({"resell_cost": round(pred_price, 2)})
    except Exception as e:
        logger.error(f"Error in resell prediction: {str(e)}")
        return jsonify({"error": "Invalid input or model not loaded"}), 400

@app.route('/frontend', methods=['GET', 'POST'])
def washing_machines():
    result = None
    companies = wm_label_encoders['Company'].classes_ if wm_df is not None and 'Company' in wm_label_encoders else []
    types = wm_label_encoders['Type'].classes_ if wm_df is not None and 'Type' in wm_label_encoders else []
    problems = wm_label_encoders['Problem'].classes_ if wm_df is not None and 'Problem' in wm_label_encoders else []
    if request.method == 'POST':
        company = request.form['company']
        type_ = request.form['type']
        year = int(request.form['year'])
        problem = request.form['problem']
        result = predict_washing_machine(company, type_, year, problem)
    return render_template('frontend.html', result=result, companies=companies, types=types, problems=problems)

@app.route('/refrigerator', methods=['GET', 'POST'])
def refrigerator():
    result = None
    companies = refrig_le_company.classes_ if refrig_df is not None else []
    types = refrig_le_type.classes_ if refrig_df is not None else []
    problems = refrig_le_problem.classes_ if refrig_df is not None else []
    if request.method == 'POST':
        company = request.form['company']
        type_ = request.form['type']
        year = int(request.form['year'])
        problem = request.form['problem']
        result = predict_refrigerator(company, type_, year, problem)
    return render_template('refrigerator.html', result=result, companies=companies, types=types, problems=problems)

@app.route('/printers', methods=['GET', 'POST'])
def printer():
    result = None
    brands = printer_le_brand.classes_ if printer_df is not None else []
    types = printer_le_type.classes_ if printer_df is not None else []
    problems = printer_le_problem.classes_ if printer_df is not None else []
    if request.method == 'POST':
        brand = request.form['brand']
        type_ = request.form['type']
        year = int(request.form['year'])
        problem = request.form['problem']
        result = predict_printer(brand, type_, year, problem)
    return render_template('printers.html', result=result, brands=brands, types=types, problems=problems)

@app.route('/appliance', methods=['GET', 'POST'])
def appliance():
    result = None
    available_subtypes = []
    available_problems = []
    appliance_types = appliance_le_appliance.classes_ if appliance_df is not None else []
    if request.method == 'POST':
        appliance_type = request.form['appliance_type']
        subtype = request.form['subtype']
        year = int(request.form['year'])
        problem = request.form['problem']
        result = predict_appliance(appliance_type, subtype, year, problem)
    appliance_type = request.form.get('appliance_type', 'TV')
    available_subtypes = subtype_categories.get(appliance_type, subtype_categories['TV'])
    available_problems = problem_categories.get(appliance_type, {}).get(
        request.form.get('subtype', available_subtypes[0]), 
        problem_categories[appliance_type][available_subtypes[0]]
    )
    return render_template('appliance.html', result=result, 
                          subtype_categories=subtype_categories, 
                          problem_categories=problem_categories, 
                          available_subtypes=available_subtypes, 
                          available_problems=available_problems,
                          appliance_types=appliance_types)

@app.route('/mobile_phone', methods=['GET', 'POST'])
def mobile_phone():
    result = None
    brands = mobile_label_encoders['Brand'].classes_ if mobile_df is not None and 'Brand' in mobile_label_encoders else []
    features = mobile_label_encoders['Feature'].classes_ if mobile_df is not None and 'Feature' in mobile_label_encoders else []
    oses = mobile_label_encoders['OS'].classes_ if mobile_df is not None and 'OS' in mobile_label_encoders else []
    problems = mobile_label_encoders['Problem'].classes_ if mobile_df is not None and 'Problem' in mobile_label_encoders else []
    if request.method == 'POST':
        brand = request.form['brand']
        feature = request.form['feature']
        os = request.form['os']
        year = int(request.form['year'])
        problem = request.form['problem']
        result = predict_mobile_phone(brand, feature, os, year, problem)
    return render_template('mobile_phone.html', result=result, brands=brands, features=features, oses=oses, problems=problems)

@app.route('/batteries', methods=['GET', 'POST'])
def batteries():
    result = None
    battery_types = battery_df['Battery_Type'].unique() if battery_df is not None else []
    brands = battery_df['Brand'].unique() if battery_df is not None else []
    problems = battery_df['Problem'].unique() if battery_df is not None else []
    if request.method == 'POST':
        battery_type = request.form['battery_type']
        brand = request.form['brand']
        problem = request.form['problem']
        repairable, action, cost = predict_battery(battery_type, brand, problem)
        result = {'repairable': repairable, 'action': action, 'cost': cost}
    return render_template('batteries.html', result=result, battery_types=battery_types, brands=brands, problems=problems)



@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {str(e)}")
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)