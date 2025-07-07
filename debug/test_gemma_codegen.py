#%%
import requests
import os
from niceode.mlflow_utils import get_class_source_without_docstrings
from niceode.diffeqs import OneCompartmentAbsorption, OneCompartmentBolus_CL
# Get the URL from the environment variable set in docker-compose
GEMMA_URL = os.getenv("GEMMA_API_URL", "http://localhost:8000")

example_class = get_class_source_without_docstrings(OneCompartmentAbsorption)
update_class = get_class_source_without_docstrings(OneCompartmentBolus_CL)


prompt = {"problem_statement":f"""
The following task will require synthesis of code and deep understanding of ordinary differential equations (ODEs) used in pharamcokinetic modeling.
The user will perform the modeling of ODE parameters and utilize tools like diffrax and scipy to solve the ODE. The task at hand consists of carefully 
examining the example python class representing a one-compartment ODE model, understanding how its' methods relate to each other and the underlyling ODE, 
then sythesizing the conceptual framework demonstrated in the example class by adding analagous methods to the class requiring update. A correct understanding 
of WHY the example classes methods are defined the way they are will lead to a correct sythesis of the new methods. Unless the two classes
describe the exact same ODE it is unlikely that the methods from the example class can be directly copied to the to-be-updated class. In particular the non-dimensional 
and hybrid dimensional methods will need to be derived from scratch following a bioengineering+pharamcology assesment of the meaning and units of the ode method
for the to-be-updated class.

Here is an example Python class for a one-compartment model with absorption:
---
{example_class}
---

And here is the class that needs to be updated, which describes a one-compartment bolus model:
---
{update_class}
---

Begin by setting the stage for success, first things first. 
What knowledge will DiffeqGemma need to complete the task? What are the 'sharp edges' of the problem? 
Where could DiffeqGemma go down the wrong path?
How can we proactively avoid the wrong path and set ourselves on a robust path towards the solution?

With planning completed: For the example class describe each method, what is the purpose of each method and what can we infer about the units
given the structure of the code?
Which methods are present in the example class but not the to-be-updated class? For the present methods what is the purpose of each method and
what can we infer about the units given the structure of the code? What about the to-be-added methods, what is the purpose of each method and 
what can we say about the units?
"""}
prompt = str(prompt)

#%%

txt2 = """Based on the provided code and DiffeqGemma's knowledge of pharmacokinetics, begin by describing the ODEs represented by these classes.
What does each ODE describe? What are the critical parameters? How do the different methods relate to each other? Why would we want to non-dimensionalize an ODE?
In the example class, what is the logic of the non-dimensionalization? How can we generally apply the idea of non-dimensionalization. 
"""    


response = requests.post(f"{GEMMA_URL}/generate", json={"text": prompt})

#print(response.json()["generated_text"])
#%%
def improve_clarify_loop(txt_in):
    
    critique_template = """The ***Text For Evaluation*** is a response from DiffeqGemma engaged in preparation
    to update a python class describing OneCompartmentBolus_CL given an example class OneCompartmentAbsorption.
    Evaluate and critique the response. Did DiffeqGemma correctly respond? Could the response be improved or clarified? 
    If improvements or clarifications are required or beneficial add a brief, clear sections titled 
    'Evaluation' and 'Improvement/Clarification' to set DiffeqGemma straight and ensure it is prepared to move on. 
    Assign a rating between 0 and 10 (formatted: 00, 01, 02 . . . 09, 10) to the reponse.
    0 being the worst possible reponse, 10 being the best using the template
    '**Response Quality:__**'. 
    The ***Text For Evaluation***:
    ---------
    """
    
    critique_prompt = f"""{critique_template}
    {txt_in}
    ---------
    """
    
    response = requests.post(f"{GEMMA_URL}/generate", json={"text": critique_prompt})
    critique_reply = response.json()['generated_text']
    while "**Response Quality:10**" not in critique_reply:
        fix_ur_work_prompt = f"""The ***Evaluation Improvement and Clarification*** is a chain of thought from DiffeqGemma.
        Read and internalize the Evaluation provided, then comply with the Improvement/Clarification.
        The ***Evaluation Improvement and Clarification***:
        ---------
        {critique_reply}
        ---------
        """
        response = requests.post(f"{GEMMA_URL}/generate", json={"text": fix_ur_work_prompt})
        updated_response = response.json()['generated_text']
        
        critique_prompt = f"""{critique_template}
        {updated_response}
        ---------
        """
        
        response = requests.post(f"{GEMMA_URL}/generate", json={"text": critique_prompt})
        critique_reply = response.json()['generated_text']
        
    return critique_reply
        
        
tmp = improve_clarify_loop(response.json()['generated_text'])     
#%%     


prompt = " ".join([
    "The following is a chain of thought from DiffeqGemma", 
    "------",
    response.json()['generated_text'], 
    "------",
    "DiffeqGemma is ready to describe non-dimensionalization and how to apply the concept to the diffeqs discussed, please proceed"
])

response = requests.post(f"{GEMMA_URL}/generate", json={"text": prompt})

#%%
prompt = " ".join([
    "The following is a chain of thought from DiffeqGemma", 
    "Evaluate and critique the dialogue. Did DiffeqGemma correctly respond following critique? Could the response be improved or clarified?", 
    "If so, add a brief, clear section to set DiffeqGemma straight and ensure it is prepared to move on to updating the python classes",
    "If DiffeqGemma is ready to proceed reply: READYTOPROCEED2",
    "The text requiring evaluation: ",
    response.json()['generated_text']
])

response6 = requests.post(f"{GEMMA_URL}/generate", json={"text": prompt6})


promptn = ["Carefully consider the structure and purpose of the following python class for use with an ODE solver like scipy's `solve_ivp`: ",
          f"```python {example_class} ```",
          "Using that class as a conceptual reference, add a correctly derived `nondim_diffrax_ode` method to the following class: ",
          update_class,
          "It is not sufficent to copy the method from the example class, you must derive the new method using your knowledge of calculus and the ODE defined in the to-be-updated class's `ode` method."]



#nondim_diffrax_ode
#nondim_to_concentration
#get_nondim_defs
#get_nondim_time
# %%
