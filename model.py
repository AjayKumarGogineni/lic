import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import json
filename = 'description.json'
with open(filename, 'r', encoding='utf-8') as f:
    plan_description = json.load(f)
# plan_description

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    
    # Convert date columns to datetime
    date_columns = ['Commence Date', 'Maturity Date', 'Policy Holder Birth Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Calculate the age at policy commencement
    df['Age at Commencement'] = df['Commence Date'].dt.year - df['Policy Holder Birth Date'].dt.year

    # Calculate years left in the policy
    df['Years Left in Plan'] = df['Maturity Date'].dt.year - datetime.now().year

    df['Commence Year'] = df['Commence Date'].dt.year
    df['Commence Month'] = df['Commence Date'].dt.month
    df['Maturity Year'] = df['Maturity Date'].dt.year
    df['Maturity Month'] = df['Maturity Date'].dt.month

    df['Years in Plan'] = datetime.now().year - df['Commence Date'].dt.year
    
    # Calculate age at policy commencement
    # df['Age at Commencement'] = (df['Commence Date'] - df['Policy Holder Birth Date']).astype('<m8[Y]')
    
    # Assign age group
    df['Age Group'] = pd.cut(df['Age at Commencement'], 
                             bins=[-np.inf, 13, 24, 40, 60, np.inf], 
                             labels=['0-13', '14-24', '25-40', '41-60', '60+'])
    
    # Calculate additional features
    current_date = datetime.now()
    # df['Years in Plan'] = (current_date - df['Commence Date']).astype('<m8[Y]')
    # df['Years Left in Plan'] = (df['Maturity Date'] - current_date).astype('<m8[Y]')
    df['Total Premium Paid'] = df['Premium Paying Term'] * df['Term']  # This is an estimate, adjust as needed
    
    return df

# Feature engineering
def engineer_features(df):
    # One-hot encode categorical variables
    categorical_cols = ['Age Group', 'Policy Plan']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Select features for the model
    features = ['Age at Commencement', 'Term', 'Premium Paying Term', 'Sum Assured', 
                'Years in Plan', 'Years Left in Plan', 'Total Premium Paid'] + \
               [col for col in df_encoded.columns if col.startswith(tuple(categorical_cols))]
    
    X = df_encoded[features]
    y = df_encoded['Plan Name']
    
    return X, y

def engineer_user_features(user_data):
    """
    Process user data to create features matching those used in the ML model.
    
    :param user_data: dict, contains user information
    :return: list, engineered features for the user
    """
    features = []
    
    # Add basic features
    features.append(user_data['Age at Commencement'])
    features.append(user_data.get('Term', 0))  # Default to 0 if not provided
    features.append(user_data.get('Premium Paying Term', 0))
    features.append(user_data.get('Sum Assured', 0))
    
    # Calculate or estimate additional features
    current_date = datetime.now()
    policy_start_date = user_data.get('Policy Start Date', current_date)
    years_in_plan = (current_date - policy_start_date).days / 365.25 if isinstance(policy_start_date, datetime) else 0
    features.append(years_in_plan)
    
    policy_end_date = user_data.get('Policy End Date', current_date)
    years_left_in_plan = (policy_end_date - current_date).days / 365.25 if isinstance(policy_end_date, datetime) else 0
    features.append(max(0, years_left_in_plan))  # Ensure non-negative
    
    total_premium_paid = user_data.get('Total Premium Paid', 0)
    features.append(total_premium_paid)
    
    # One-hot encode categorical variables
    age_groups = ['0-13', '14-24', '25-40', '41-60', '60+']
    for group in age_groups:
        features.append(1 if user_data.get('Age Group') == group else 0)
    
    policy_plans = ['Plan A', 'Plan B', 'Plan C', 'Plan D']  # Add all possible plans
    for plan in policy_plans:
        features.append(1 if user_data.get('Current Policy Plan') == plan else 0)
    
    return features

# Train machine learning model
def train_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, X_test_scaled, y_test

def load_llm_model():
    return pipeline("text-generation", model="gpt2")
    model_name = "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Llama-2-7b-chat-hf"  # or another Llama 2 variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HUGGING_FACE_TOKEN = "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"
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.float16, 
                                                 device_map="auto",
                                                 token=HUGGING_FACE_TOKEN)
    return model, tokenizer

# Generate text using Llama 2
def generate_llm_recommendation(model, 
                                tokenizer,
                                prompt, 
                                max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.3,
            # do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# Main function
def main():
    file_path = 'data/Imagic Lic Data.xlsx'
    df = load_and_preprocess_data(file_path)
    X, y = engineer_features(df)
    ml_model, scaler = train_ml_model(X, y)
    llm_model, llm_tokenizer = load_llm_model()
    
    # Example user data
    user_data = {
        'Age': 35,
        'Age Group': '25-40',
        'Term': 20,
        'Premium Paying Term': 15,
        'Sum Assured': 500000,
        'Policy Start Date': datetime(2020, 1, 1),
        'Policy End Date': datetime(2040, 1, 1),
        'Total Premium Paid': 100000,
        'Current Policy Plan': 'Plan B'
    }
    plan_descriptions = plan_description
    # Load plan descriptions (you'll need to create this dictionary based on your plan documents)
    # plan_descriptions = {
    #     "Amritbaal": "LIC's Amritbaal Term Age Minimum Sum Assured 5 0 (30 Days) Maximum 2,00,000 25 13 Minimum PPT: 1 Minimum Maturity Age: 18...",
    #     # Add other plan descriptions
    # }
    
    recommended_plan, llm_recommendation = recommend_plan(ml_model, scaler, llm_model, llm_tokenizer, user_data, plan_descriptions)
    
    print(f"Recommended Plan: {recommended_plan}")
    print(f"Personalized Recommendation: {llm_recommendation}")

# main()

# Train the model
file_path = 'data/Imagic Lic Data.xlsx'
df = load_and_preprocess_data(file_path)
df = df[:1000]
X, y = engineer_features(df)
ml_model, scaler, X_test_scaled, y_test = train_ml_model(X, y)

# GPT2 code

llm_model = load_llm_model()

def recommend_plan_gpt(model, scaler, llm, user_data, plan_descriptions):
    user_features = engineer_user_features(user_data)
    user_features_scaled = scaler.transform([user_features])
    
    recommended_plan = model.predict(user_features_scaled)[0]
    plan_description = plan_descriptions.get(recommended_plan, "Plan description not available.")
    
    # Use LLM to generate a personalized recommendation
    prompt = f"Given the user's age group of {user_data['Age Group']} and the recommended plan '{recommended_plan}', provide a personalized insurance recommendation. Plan details: {plan_description}"
    llm_recommendation = llm(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    
    return recommended_plan, llm_recommendation

# Example user data
index = 0
user_data = X_test_scaled[0]#X.iloc[index:index+1].to_dict(orient='records')[0]
# {
# 'Age': 35,
# 'Age Group': '25-40',
# 'Term': 20,
# 'Premium Paying Term': 15,
# 'Sum Assured': 500000,
# 'Policy Start Date': datetime(2020, 1, 1),
# 'Policy End Date': datetime(2040, 1, 1),
# 'Total Premium Paid': 100000,
# 'Current Policy Plan': 'Plan B'
# }

# Load plan descriptions (you'll need to create this dictionary based on your plan documents)
plan_descriptions = plan_description

# Recommend plan using ML model and LLM
def recommend_plan(ml_model, scaler, llm_model, llm_tokenizer, user_data, plan_descriptions):
    # user_features = engineer_user_features(user_data)
    # user_features_scaled = scaler.transform([user_features])
    user_features_scaled = user_data
    recommended_plan = ml_model.predict(user_features_scaled)[0]
    plan_description = plan_descriptions.get(recommended_plan, "Plan description not available.")
    
    # Use Llama 2 to generate a personalized recommendation
    prompt = f"""
    Given the following information:
    - User's age group: {user_data['Age Group']}
    - Recommended insurance plan: '{recommended_plan}'
    - Plan details: {plan_description}

    Provide a concise and personalized insurance recommendation for the user. The recommendation should be friendly, informative, and highlight the key benefits of the plan for someone in this age group.
    """
    llm_recommendation = generate_llm_recommendation(llm_model, llm_tokenizer, prompt)
    
    return recommended_plan, llm_recommendation

recommended_plan, llm_recommendation = recommend_plan_gpt(ml_model, scaler, llm_model, user_data, plan_descriptions)

print(f"Recommended Plan: {recommended_plan}")
print(f"Personalized Recommendation: {llm_recommendation}")

# main()