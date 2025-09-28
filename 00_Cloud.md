# 🧠 Generative AI + AWS Interview Questions (Student-Style Answers)

## 1. What are the key differences between traditional ML models and generative AI models?
Traditional ML models **predict or classify** (like detecting spam). Generative AI models **create new content**, such as text, code, or images.

---

## 2. How do transformer models work, and why are they suitable for generative AI?
Transformers use **self-attention** to understand word relationships. They are good for GenAI because they can handle long sequences and generate meaningful content.

---

## 3. What are the typical use cases of generative AI in the cloud?
- Chatbots  
- Text summarization  
- Image generation  
- Code generation  
- Personalized recommendations  
- Voice assistants

---

## 4. How would you deploy a generative AI model using AWS services?
Use **SageMaker** to train and host the model. Or use **Bedrock** to access pre-trained models via API. **S3** stores data, and **Lambda** or **API Gateway** connects it to apps.

---

## 5. How does Amazon Bedrock simplify the use of foundation models?
It lets us access top models (like from Anthropic, Meta, etc.) **without training** or managing infrastructure. Just use an API.

---

## 6. Compare SageMaker with Bedrock. When would you use one over the other?
- **Bedrock** is for using or fine-tuning pre-built models.  
- **SageMaker** is for training custom models or doing full control ML.

---

## 7. How do you optimize cost and performance when training LLMs on AWS?
Use:
- **Spot instances**  
- **Mixed-precision training**  
- **Efficient data loaders**  
- **Right instance types** (like P4, P5)  
- **Checkpoints** to avoid re-training

---

## 8. Which instance types are suitable for GenAI on AWS?
- **Training:** P4, P5 (GPU-heavy)  
- **Inference:** Inf2 (optimized for inference)

---

## 9. How would you set up distributed training for large models in SageMaker?
Use **SageMaker Distributed Training** with **data parallelism** or **model parallelism**, and multiple GPU instances.

---

## 10. What are the key security considerations for GenAI on AWS?
- Use **IAM roles**  
- **Encrypt data** in transit and at rest  
- Log access with **CloudTrail**  
- Use **private endpoints** if needed

---

## 11. How do you ensure data privacy and compliance in Bedrock or SageMaker?
- Don’t store sensitive prompts  
- Use **VPC + IAM policies**  
- Use **KMS** for encryption  
- Follow **AWS compliance best practices**

---

## 12. Suppose your client wants a chatbot powered by GenAI that can access private enterprise documents. How would you design and deploy it on AWS?
1. Store documents in **S3**  
2. Use **Kendra** or **OpenSearch** for retrieval  
3. Use **Bedrock** for LLM generation (RAG setup)  
4. Serve via **API Gateway + Lambda**

---

## 13. How would you fine-tune an open-source LLM on AWS?
1. Upload data to **S3**  
2. Use **SageMaker** with Hugging Face/TensorFlow  
3. Choose **GPU instance** (like P4d)  
4. Train, evaluate, and deploy model endpoint

---

## 14. How do you handle prompt engineering in Bedrock and check output quality?
Write and test different prompts.  
Evaluate output using:
- Manual review  
- **BLEU**, **ROUGE**, or **Factuality** checks

---

## 15. How would you implement CI/CD for GenAI with SageMaker?
1. Use **CodePipeline + CodeBuild**  
2. Automate training, testing, and deployment  
3. Register models in **SageMaker Model Registry**  
4. Deploy to **SageMaker Endpoint**

---

## 16. What are the limitations of GenAI, and how does AWS help?
Limitations:
- Bias
- Hallucination
- Cost

AWS helps with:
- Monitoring (CloudWatch)
- Guardrails (Bedrock)
- Optimization tools (e.g., SageMaker Neo)

---

## 17. What is RAG, and how do you implement it on AWS?
**RAG = Retrieval-Augmented Generation**

Steps:
1. Search relevant data using **Kendra/OpenSearch**  
2. Pass results into **Bedrock** LLM  
3. Get a more accurate and grounded response

---
## 18. Service of AWS.

```python
| Category             | AWS Service         | Purpose in GenAI Projects                                      |
|----------------------|---------------------|----------------------------------------------------------------|
| AI / ML              | Amazon Bedrock      | Use foundation models via API                                  |
|                      | Amazon SageMaker    | Train, fine-tune, and deploy custom models                     |
|                      | Amazon Comprehend   | NLP for text understanding                                     |
|                      | Amazon Transcribe   | Convert speech to text                                         |
|                      | Amazon Polly        | Convert text to speech                                         |
|                      | Amazon Rekognition  | Analyze images and videos                                      |
|                      | Amazon Lex          | Build conversational chatbots                                  |
| Compute              | EC2                 | Run GenAI workloads manually                                   |
|                      | SageMaker Studio    | IDE for managing training and pipelines                        |
|                      | ECS / EKS           | Run GenAI apps with containers / Kubernetes                    |
|                      | Lambda              | Trigger functions (e.g., prompt processing)                    |
| Storage              | Amazon S3           | Store training data, models, and outputs                       |
|                      | Amazon EFS / FSx    | Shared file storage for model training                         |
|                      | Amazon Redshift     | Analyze structured data for training                           |
|                      | Amazon Athena       | Query data in S3 for preprocessing                             |
| Search / RAG         | Amazon Kendra       | Intelligent search across documents                            |
|                      | Amazon OpenSearch   | Search and vector retrieval for RAG                            |
|                      | Amazon Neptune      | Knowledge graphs for advanced GenAI                            |
| Security             | IAM                 | Manage access to resources                                     |
|                      | AWS KMS             | Encrypt data and model files                                   |
|                      | AWS Config          | Ensure compliance of infrastructure                            |
|                      | AWS CloudTrail      | Log API activity                                               |
| Monitoring           | Amazon CloudWatch   | Monitor training and inference metrics                         |
|                      | AWS X-Ray           | Trace GenAI app requests                                       |
| MLOps / DevOps       | CodePipeline        | Automate model deployment                                      |
|                      | CodeBuild           | Build and test model components                                |
|                      | SageMaker Pipelines | CI/CD for ML workflows                                         |
|                      | Step Functions      | Orchestrate GenAI workflows                                    |
```
