import json
import ollama

def process_policy(policy_details):
    prompt = """
    Given the insurance policy details below, please:
    1. Extract the exact name of the policy.
    2. Summarize the policy extensively, write all key features and benefits. do not miss any numerical details.
    3. Return the result in JSON format with keys "name" and "policy_summary".

    Policy details:

    {policy_details}

    Your response should be in the following JSON format:
    {{
        "name": "Exact Policy Name",
        "policy_summary": "Extensive policy summary with all key features and benefits."
    }}

    Examples:

    Input:
    LIC's Jeevan Labh (Plan 836)
    A non-linked, with-profits, limited premium payment endowment assurance plan...

    Output:
    {{
        "name": "LIC's Jeevan Labh",
        "policy_summary": "A limited premium payment endowment plan offering life coverage and maturity benefits with profit sharing......."
    }}

    If you are unable to extract the exact policy name, your output should be:
    {{
        "name": "Unknown",
        "policy_summary": "Policy Summary"
    }}

    Please respond only with the JSON for the given policy, no additional text.
    """

    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt.format(policy_details=policy_details),
        },
    ])
    
    return response['message']['content']

# Read the original JSON file
with open('description.json', 'r', encoding='utf-8') as f:
    policies = json.load(f)

summarized_policies = {}

# Process each policy
for policy_key, policy_details in policies.items():
    print(f"Processing policy: {policy_key}")
    
    result = process_policy(policy_details)
    print(f"LLM response received for: {policy_key}")
    
    try:
        parsed_result = json.loads(result)
        summarized_policies[policy_key] = parsed_result
        print(f"Successfully processed: {policy_key}")
    except json.JSONDecodeError:
        print(f"Error parsing LLM response for: {policy_key}. Skipping this policy.")
    
    print()  

# Write the summarized policies to a new JSON file
with open('new_summarized_policies.json', 'w') as f:
    json.dump(summarized_policies, f, indent=2)

print("Summarization complete. Results saved in 'summarized_policies.json'")