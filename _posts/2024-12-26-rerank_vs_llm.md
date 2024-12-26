---
title:  "LLMs vs Rerankers for Intent Routing"
mathjax: true
layout: post
categories: media
---

Intent routing is essential for many Generative AI (GenAI) applications, enabling systems to accurately interpret user queries and route them to the appropriate actions. With the rise of Large Language Models (LLMs), their flexibility and contextual understanding have made them a go-to choice for intent classification tasks. However, embedding-based rerankers offer a compelling alternative, delivering high accuracy with significantly lower computational costs and latency.

This analysis compares the performance of a reranker model (`BAAI/bge-reranker-large`) versus a state-of-the-art LLM (`claude-3-haiku-20240307`) for intent routing, focusing on their trade-offs in efficiency, accuracy, and suitability for real-world use cases.

---



## Why Compare Rerankers and LLMs?

In agent-based GenAI workflows, an LLM typically serves as a router to classify intents and guide users to the right workflows. While LLMs excel in nuanced reasoning and human-like interaction, their reliance for every task—intent classification included—can be resource-intensive.

Rerankers, on the other hand, are specialized embedding models that compare intents (sentences) with an anchor (query) by reranking possible responses (intents/sentences). They stand out for their speed, cost-effectiveness, and scalability, making them ideal for real-time applications.

## The Setup - Intents and Descriptions

Below is the list of possible intents along with their descriptions:

1. Health Monitoring:
    - Track daily steps, heart rate, sleep patterns, and other vital signs using wearable device data.
    - Monitor blood sugar levels for diabetic users through connected glucose monitors.
2. Medication Management:
    - Remind users when to take medications based on prescriptions or user input.
    - Alert users about potential drug interactions or side effects.
3. Diet and Nutrition:
    - Analyze food intake by integrating with dietary tracking apps.
    - Suggest meal plans or recipes based on dietary restrictions, allergies, or health goals like weight loss or managing cholesterol.
4. Fitness Guidance:
    - Recommend personalized workout plans based on fitness levels, goals, and past activity data.
    - Provide real-time feedback during exercises or suggest modifications for injuries.
5. Appointment Scheduling:
    - Schedule, remind, and reschedule medical appointments.
    - Integrate with calendar apps to ensure no conflicts with other commitments.
6. Health Insights and Alerts:
Generate health reports summarizing activity, diet, sleep, and health metrics.
    - Alert users to any anomalies in health data that might require medical attention.
7. Mental Health Support:
    - Offer meditation sessions or mindfulness exercises.
    - Provide cognitive behavioral therapy techniques or connect users to mental health resources if needed.
8. Privacy and Data Security:
    - Ensure all data handling complies with privacy laws like HIPAA or GDPR.
    - Encrypt personal health information to protect user data.

---

### Prompt for Intent Routing

The LLM will analyze the provided user request and classify it into one of the above intents using the following prompt:  

**Prompt:** 

```
You will be acting as a health monitoring request classification system. Your task is to analyze user health-related requests and output the appropriate classification intent, along with your reasoning.

Here is the health request you need to classify:

<request>{user_request}</request>

Please carefully analyze the above request to determine the user's core health needs and concerns.

First, write out your reasoning and analysis of how to classify this request inside <reasoning> tags.

Then, output the appropriate classification label for the request inside an <intent> tag. The valid intents are:
<intents>
<intent>Health Monitoring: Track vital signs, steps, sleep patterns, and blood sugar levels via wearables</intent>
<intent>Medication Management: Medication reminders and interaction alerts</intent>
<intent>Diet and Nutrition: Food tracking, meal planning, and dietary recommendations</intent>
<intent>Appointment Scheduling: Medical appointment scheduling and calendar integration</intent>
<intent>Health Insights and Alerts: Health reports and anomaly detection</intent>
<intent>Fitness Guidance: Personalized workout plans and exercise feedback</intent>
<intent>Mental Health Support: Meditation, mindfulness, and therapy resources</intent>
</intents>

A request may have ONLY ONE applicable intent. Choose the intent that best addresses the user's primary need.

Example classification:
<request>I need help tracking my daily blood sugar readings and getting alerts if they go too high.</request>

<reasoning>
The user is specifically asking about monitoring blood sugar levels and receiving alerts based on those readings. While this could involve health insights, the primary focus is on ongoing health monitoring of a specific vital sign.</reasoning>

<intent>Health Monitoring</intent>
```

## Experiment

I evaluated the performance of the reranker and the LLM across diverse healthcare-related queries.

**Breakdown:**

- **Reranker:** Generates similarity scores between a query and predefined intents, leveraging [`text-embedding-inference`](https://github.com/huggingface/text-embeddings-inference) from Hugging Face. I conducted two experiments: one reranks based on the query and intent only (`Reranker Time`), and another one with the intent and corresponding description (`Reranker Time w. Description`).
- **LLM:** Provides detailed reasoning alongside intent classification using the [Claude 3 Haiku model from Anthropic](https://www.anthropic.com/news/claude-3-family).

**Metrics and Dataset**

- **Inference Time:** To measure speed and efficiency. Spoiler: rerankers are a lot faster.
- **Accuracy:** Both models correctly classified intents across all queries.
- **Output Quality:** Confidence scores for the reranker and interpretive reasoning for the LLM.

| **Query**                                                                 | **Correct Intent**          | **Reranker Time w. Description** | **Reranker Time** | **LLM Time** | **Reranker Intent** | **LLM Intent** | 
|---------------------------------------------------------------------------|-----------------------------|----------------------------|----------------------------|--------------|-------------------------------|--------------------------|
| How many steps did I walk today?                                          | Health Monitoring           | 0.590 seconds             | 0.295 seconds             | 1.967318     | Health Monitoring            | Health Monitoring       |
| Can you remind me to take my blood pressure medication at 8 PM?           | Medication Management       | 0.305 seconds             | 0.321 seconds             | 1.312687     | Medication Management        | Medication Management   |
| What should I eat for dinner if I want to keep my cholesterol low?        | Diet and Nutrition          | 0.324 seconds             | 0.450 seconds             | 1.863424     | Diet and Nutrition           | Diet and Nutrition      |
| Can you book an appointment with my cardiologist next Thursday?           | Appointment Scheduling      | 0.470 seconds             | 0.286 seconds             | 0.990974     | Appointment Scheduling       | Appointment Scheduling  |
| Show me my sleep pattern from last week.                                  | Health Monitoring           | 0.288 seconds             | 0.278 seconds             | 1.25709      | Health Monitoring            | Health Monitoring       |
| What exercises can I do with my current back injury?                      | Fitness Guidance            | 0.318 seconds             | 0.313 seconds             | 1.339999     | Fitness Guidance             | Fitness Guidance        |
| Is there any interaction between my new allergy medication and my heart medication? | Medication Management       | 0.328 seconds             | 0.302 seconds             | 1.025169     | Medication Management        | Medication Management   |
| I've been feeling stressed; can you suggest some relaxation techniques?   | Mental Health Support       | 0.479 seconds             | 0.314 seconds             | 1.150373     | Mental Health Support        | Mental Health Support   |

### Observations

1. **Processing Time**:
   - **Reranker Performance**: The reranker processes queries significantly faster compared to the LLM, with times ranging from **0.278 to 0.590 seconds** depending on whether descriptions are included. Intent-only processing is consistently quicker.
   - **LLM Performance**: The LLM takes longer for inference, with processing times between **0.990 and 1.967 seconds**, reflecting the heavier computational load of large language models.
   - Overall, the LLM requires approximately **3-5x more time** than the reranker on average and is much more expensive, since the reranker is self-hosted and can run locally.

2. **Results**:
   - Both methods accurately classify intents across diverse queries.
   - The reranker provides confidence scores that reflect its decision-making process, while the LLM outputs detailed reasoning that offers insights into its interpretation of the query.
   - The descriptions didn't have any impact in the rerankers intent classification accuracy. Discarding descriptions make for inference and simpler setups.

3. **Suitability**:
   - **Reranker**: Best suited for **real-time, high-throughput systems** where processing speed is critical. Confidence scores offer a reliable and lightweight evaluation mechanism.
   - **LLM**: Excels in **complex or ambiguous queries** where detailed reasoning or interpretability is required, such as in **clinical decision support or debugging scenarios**.

4. **Trade-Offs**:
   - **Reranker** prioritizes efficiency and scalability but lacks the depth of explanation provided by the LLM.
   - **LLM** provides rich, interpretative reasoning but at the cost of increased processing time, making it better suited for scenarios where time sensitivity is less critical. On the other hand, prompt engineering is required and the setup is much more complex.

## Additional Insights

### Reranker Analysis

Below is a detailed breakdown of the primary tasks associated with each query, along with the ranked relevance of the intents based on the reranker's output. The time taken for each query is also provided, distinguishing between the time taken for intents with and without description.

| **Query** | **Primary Task** | **Rank 1** | **Rank 2** | **Rank 3** | **Rank 4** | **Rank 5** | **Rank 6** | **Rank 7** | **Time Taken (with desc./without desc.)** |
|-----------|------------------|------------|------------|------------|------------|------------|------------|------------|---------------|
| **How many steps did I walk today?** | Health Monitoring | Health Monitoring (0.0046) | Fitness Guidance (0.0025) | Health Insights and Alerts (0.0003) | Diet and Nutrition (0.0003) | Mental Health Support (0.0001) | Appointment Scheduling (0.00008) | Medication Management (0.00008) | 0.589s / 0.295s |
| **Can you remind me to take my blood pressure medication at 8 PM?** | Medication Management | Medication Management (0.0061) | Appointment Scheduling (0.0002) | Health Insights and Alerts (0.0001) | Health Monitoring (0.0001) | Diet and Nutrition (0.00008) | Mental Health Support (0.00008) | Fitness Guidance (0.00008) | 0.305s / 0.321s |
| **What should I eat for dinner if I want to keep my cholesterol low?** | Diet and Nutrition | Diet and Nutrition (0.0046) | Health Insights and Alerts (0.0002) | Health Monitoring (0.0001) | Fitness Guidance (0.00009) | Mental Health Support (0.00008) | Appointment Scheduling (0.00008) | Medication Management (0.00008) | 0.324s / 0.450s |
| **Can you book an appointment with my cardiologist next Thursday?** | Appointment Scheduling | Appointment Scheduling (0.0002) | Diet and Nutrition (0.00008) | Health Monitoring (0.00008) | Mental Health Support (0.00008) | Health Insights and Alerts (0.00008) | Medication Management (0.00008) | Fitness Guidance (0.00008) | 0.470s / 0.286s |
| **Show me my sleep pattern from last week.** | Health Monitoring | Health Monitoring (0.0059) | Health Insights and Alerts (0.0016) | Mental Health Support (0.0003) | Diet and Nutrition (0.0001) | Fitness Guidance (0.00008) | Medication Management (0.00008) | Appointment Scheduling (0.00008) | 0.288s / 0.278s |
| **What exercises can I do with my current back injury?** | Fitness Guidance | Fitness Guidance (0.0038) | Mental Health Support (0.0009) | Health Monitoring (0.00008) | Diet and Nutrition (0.00008) | Health Insights and Alerts (0.00008) | Medication Management (0.00008) | Appointment Scheduling (0.00008) | 0.318s / 0.313s |
| **Is there any interaction between my new allergy medication and my heart medication?** | Medication Management | Medication Management (0.0003) | Diet and Nutrition (0.00008) | Mental Health Support (0.00008) | Fitness Guidance (0.00008) | Health Insights and Alerts (0.00008) | Appointment Scheduling (0.00008) | Health Monitoring (0.00008) | 0.328s / 0.302s |
| **I've been feeling stressed; can you suggest some relaxation techniques?** | Mental Health Support | Mental Health Support (0.0143) | Fitness Guidance (0.0001) | Health Monitoring (0.00008) | Appointment Scheduling (0.00008) | Diet and Nutrition (0.00008) | Medication Management (0.00008) | Health Insights and Alerts (0.00008) | 0.479s / 0.314s |

### LLM Analysis

The LLM provides detailed reasoning for each query, offering insights into the decision-making process. The reasoning aligns with the correct intent classification, demonstrating the model's ability to interpret and respond to healthcare-related queries effectively.

| **Query**                                                                                  | **Intent**                                                                          | **Reasoning**                                                                                                                                                                                                                                                                                                                                                                                                                         | **Time Taken (s)** |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| How many steps did I walk today?                                                           | Health Monitoring: Track vital signs, steps, sleep patterns, and blood sugar levels via wearables | The user is asking about tracking their daily step count, which is a common metric tracked through wearable health monitoring devices like fitness trackers or smartwatches. This request is focused on monitoring physical activity levels, a key aspect of overall health and wellness. The user is not asking about medications, diet, appointments, or other specific health needs. | 1.967              |
| Can you remind me to take my blood pressure medication at 8 PM?                            | Medication Management                                                             | The user's request is specifically about setting a reminder to take their blood pressure medication at a certain time. This request is focused on medication management and compliance, rather than general health monitoring or other healthcare needs.                                                                                                                                    | 1.313              |
| What should I eat for dinner if I want to keep my cholesterol low?                         | Diet and Nutrition                                                                | The user's request is focused on finding dietary recommendations to help keep their cholesterol levels low. This indicates a need for guidance on diet and nutrition, specifically around food choices that can positively impact cholesterol levels.                                                                                                                                        | 1.863              |
| Can you book an appointment with my cardiologist next Thursday?                            | Appointment Scheduling                                                            | The request mentions the user wanting to book an appointment with their cardiologist, which indicates the user has a health concern related to their heart. The user is looking to schedule a medical appointment, which is the primary need expressed in the request.                                                                                                                     | 0.991              |
| Show me my sleep pattern from last week.                                                   | Health Monitoring: Track vital signs, steps, sleep patterns, and blood sugar levels via wearables | The request is asking to view the user's sleep pattern from the previous week. This indicates a need for ongoing health monitoring and tracking of sleep data, which is a key vital sign. The user is not requesting any specific insights, alerts, or recommendations related to their sleep, but rather just access to their historical sleep data.                                      | 1.257              |
| What exercises can I do with my current back injury?                                       | Fitness Guidance                                                                  | The user is requesting guidance on exercises they can do with a current back injury. This indicates the need for personalized fitness and rehabilitation guidance, rather than general health monitoring, medication management, diet/nutrition support, appointment scheduling, or mental health resources. The user's primary concern is finding suitable exercises to address their specific back injury. | 1.340              |
| Is there any interaction between my new allergy medication and my heart medication?        | Medication Management                                                             | The request indicates the user has a concern about potential interactions between their new allergy medication and their existing heart medication. This suggests the user's primary need is around medication management, specifically monitoring potential drug interactions.                                                                                                                   | 1.025              |
| I've been feeling stressed; can you suggest some relaxation techniques?                    | Mental Health Support                                                             | The request indicates that the user is feeling stressed and is looking for relaxation techniques. This suggests that the user's primary need is related to managing their mental health and well-being, rather than physical health monitoring, medication management, diet and nutrition, appointment scheduling, or fitness guidance. The user is seeking support for their emotional state and methods to reduce stress. | 1.150              |

## Conclusion

While LLMs are versatile and powerful, they are not a one-size-fits-all solution for GenAI tasks. Embedding-based rerankers provide a robust, efficient alternative for intent routing (classification), particularly in cost-sensitive or real-time applications. On the other hand, LLMs are a compelling alternative where interpretative reasoning is required and cost and/or increased processing time aren't a constraint.

## References

- [Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference)
- [Anthropic - Ticket Routing](https://docs.anthropic.com/en/docs/about-claude/use-case-guides/ticket-routing)
- [GitHub Repository - Reranker vs LLM](https://github.com/bernardo-sb/cmp-intent-routing)