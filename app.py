import streamlit as st
import ollama
import json
from datetime import date

# Load the policy data
with open('summarized_policies.json', 'r') as f:
    policy_data = json.load(f)

# Extract policy names for the dropdown
policy_names = [policy['name'] for policy in policy_data.values() if 'name' in policy]

def get_recommendation(user_data):
    # Prepare the prompt for the LLM
    prompt = f"""
    Based on the following user information and their existing policies, suggest a new suitable life insurance policy:
    
    User Information:
    Age: {user_data['Age']}
    Date of Birth: {user_data['DOB']}
    Budget for New Policy: ₹{user_data['Budget']}
    
    Existing Policies:
    """
    
    for policy in user_data['Policies']:
        prompt += f"""
        Plan: {policy['Plan']}
        Start Date: {policy['Start Date']}
        End Date: {policy['End Date']}
        Premium Amount: ₹{policy['Premium Amount']}
        """
    
    prompt += """
    
    Available Policies:
    """
    
    for policy_key, policy_info in policy_data.items():
        if 'Age 0 -13' not in policy_key:  # Exclude child plans
            prompt += f"""
            Name: {policy_info['name']}
            Summary: {policy_info['policy_summary']}
            """

    prompt += """
    Please suggest a new policy from the available options that would complement the user's existing portfolio and fit their current age and budget. Explain why this policy would be suitable for them.
    """

    # Call the Ollama LLM
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response['message']['content']

def main():
    st.title("Life Insurance Policy Recommender")

    # User input fields
    with st.sidebar:
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        dob = st.date_input("Date of Birth", min_value=date(1900,1,1), max_value= date(2023,1,1))
        budget = st.number_input("Budget for New Policy (₹)", min_value=0.0, step=100.0)

        # Existing policies
        st.subheader("Existing Policies")
        policies = []
        
        num_policies = st.number_input("Number of existing policies", min_value=0, step=1)
        
        for i in range(num_policies):
            st.write(f"Policy {i+1}")
            plan = st.selectbox(f"Plan {i+1}", policy_names, key=f"plan_{i}")
            start_date = st.date_input(f"Start Date {i+1}", key=f"start_{i}")
            end_date = st.date_input(f"End Date {i+1}", key=f"end_{i}")
            premium = st.number_input(f"Premium Amount {i+1} (₹)", min_value=0.0, step=100.0, key=f"premium_{i}")
            
            policies.append({
                "Plan": plan,
                "Start Date": start_date.strftime("%Y-%m-%d"),
                "End Date": end_date.strftime("%Y-%m-%d"),
                "Premium Amount": premium
            })

    if st.button("Get Recommendation"):
        user_data = {
            "Age": age,
            "DOB": dob.strftime("%Y-%m-%d"),
            "Budget": budget,
            "Policies": policies
        }

        recommendation = get_recommendation(user_data)
        st.subheader("Recommended Policy:")
        st.write(recommendation)

if __name__ == "__main__":
    main()