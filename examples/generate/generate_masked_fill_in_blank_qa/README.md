# Generate Masked Fill-in-blank QAs
In this module, we generate fill-in-blank QAs from unstructured corpora by randomly masking core entities in a knowledge graph. The key is that a rule-based validator can automatically verify the answers to these questions. For example:
> **Question:** Hematogenous long-bone osteomyelitis is an infection of the bone, primarily affecting the long bones, and often results from blood-borne pathogens. This condition is characterized by several key symptoms, including ___ and swelling. ___ is a prominent symptom in both primary and recurrent cases of hematogenous long-bone osteomyelitis, manifesting as persistent discomfort in the affected area.
> **Answer:** pain

Because the answer of these questions can be easily verified, they are well-suited for RLVR (Reinforcement Learning with Verifiable Rewards).

For more details, please see our paper "Knowledge-to-Verification: Exploring RLVR for LLMs in Knowledge-Intensive Domains". It has been accepted to the ACL 2026 Main Conference, and we will update the link soon.
