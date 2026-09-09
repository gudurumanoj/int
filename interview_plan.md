# Tower Research Capital — 3-Round Interview Prep & 2-Day Plan

## Interview Structure

| Round | Interviewer | Background | Likely Focus |
|-------|-------------|------------|--------------|
| 1 | **Sarvagya Malaviya** | Core AI/ML SWE. Ex-Minus Zero (autonomous driving), Skit.ai (conversational AI). Interests: interpretability, model compression, superposition | ML/DL theory, resume deep-dive, agentic systems from the **research & engineering** lens |
| 2 | **Neeraj Agrawal** | Linux/VMware/Cloud/OpenShift Admin at Tower Gurgaon | **Infra, deployment, MLOps, Linux**, production engineering, monitoring, cost control |
| 3 | **Satyandhra D** | SDE at Tower ~4 yrs → CMU MBA → Product role. Based abroad (US) | **Product thinking, system design at strategic level**, business impact, prioritization, cost/ROI, behavioral |

**The connecting thread**: First-round feedback highlighted **hands-on agentic AI systems**. Every interviewer will probe this from their angle.

---

## Round 1: Sarvagya — ML/DL + Resume Deep-Dive

### What to Expect

He's a researcher-engineer hybrid. Expect questions that start from your resume and drill into ML fundamentals. He'll want to see you can reason about *why* things work, not just *what* you did.

### Predicted Questions

**Resume — Fractal (Agentic Systems)**
- Walk me through the architecture of your autonomous long-horizon agent system
- How do you handle the planning component? Do you use ReAct, plan-and-execute, tree-of-thought?
- How do you evaluate an agent? What metrics do you use beyond task completion rate?
- Your first interviewer asked about hill climbing — how do you prevent the agent from getting stuck in local optima during multi-step reasoning?
- How do you design eval signals that are both reliable and cheap to compute?
- What's your approach to cost control? If an agent starts making too many LLM calls, how do you detect and mitigate that?
- How do you handle the explore/exploit tradeoff in post-training for LLMs?

**Resume — Krutrim**
- Explain the synthetic pre-training data pipeline end-to-end. How did quality evaluation work?
- How did you improve the tokenizer fertility by 39.5% over LLaMA4? What specific decisions made the difference?
- How would you adapt tokenization for time-series data? (see prep doc Section 2)
- Walk me through your DeepSpeed setup for Krutrim-2. Why ZeRO and which stage?
- What went wrong during training? How did you debug a multi-node training run?
- How did you design evals for multilingual reasoning? What's hard about evaluating across languages?

**Resume — Graviton**
- What features did you use for short-term price movement forecasting?
- How did you achieve 100x latency reduction? (Quantization? Distillation? ONNX? Feature reduction?)
- Walk me through the tradeoff: why was 5% accuracy loss acceptable?

**ML/DL Theory**
- Batch norm vs layer norm — why do transformers use layer norm? (see prep doc Section 4)
- Explain dropout — why does randomly zeroing neurons help generalization?
- Derive the PCA optimization objective. Why eigenvectors of the covariance matrix?
- Xavier vs Kaiming initialization — when would you use each?
- What's the difference between MLE and MAP? Connection to L1/L2 regularization?
- Explain the attention mechanism. What's the computational complexity? How would you reduce it?
- What is KV-cache and why does it matter for inference?
- RoPE vs learned positional embeddings — tradeoffs?

**Model Compression (his interest area)**
- Compare pruning vs quantization vs distillation — when would you use each?
- What is superposition in neural networks? (He posts about this)
- How does polysemanticity affect model interpretability?
- INT8 vs INT4 quantization — what breaks at lower precision?
- What is QLoRA and why does it work?

### How to Prepare

- Rehearse Fractal agent architecture as a 3-minute story: problem → architecture → eval → production challenges
- Rehearse Krutrim tokenizer story: what was broken → your approach → results → what you'd do differently
- Re-read prep doc sections 1-7 (you already have these)
- Review attention mechanism math: Q, K, V projections, softmax(QK^T/√d)V
- Review KV-cache, RoPE, flash attention at a high level

---

## Round 2: Neeraj — Infra / Systems / MLOps

### What to Expect

He's a Linux/Cloud admin. He'll test whether you can actually **deploy, monitor, and maintain** the ML systems you build. This round separates "I trained a model in a notebook" from "I shipped it to production." Given the JD explicitly lists "CI/CD pipelines and MLOps practices" and "Linux, SQL, Git, and Bash scripting" as requirements — expect all of these.

### Predicted Questions

**Linux Fundamentals**
- How would you debug a process that's consuming too much memory on a GPU training node?
- Explain file permissions in Linux. What does `chmod 755` mean?
- How would you set up a cron job to retrain a model every Sunday at 2 AM?
- What's the difference between a process and a thread? How does this relate to Python's GIL?
- How would you find which process is using port 8080? (`lsof -i :8080` or `netstat -tlnp`)
- Explain what happens when you type `ssh user@server` and press enter
- How would you monitor GPU utilization on a multi-GPU node? (`nvidia-smi`, `gpustat`)
- What's a zombie process? How do you clean them up?
- How would you transfer a 50GB model checkpoint between servers? (`rsync`, `scp`, considerations)

**Docker & Containerization**
- Write a Dockerfile for serving a PyTorch model via FastAPI
- What's the difference between `CMD` and `ENTRYPOINT`?
- How would you reduce Docker image size for an ML serving container?
- Multi-stage builds — when and why?
- How do you pass GPU access to a Docker container? (`--gpus all`, nvidia-docker runtime)
- How would you handle model versioning in Docker? (Model baked in vs mounted volume?)
- Docker Compose vs Kubernetes — when would you use each?

**CI/CD for ML**
- Describe a CI/CD pipeline for an ML model going from training to production
- How do you version your data, code, and models together?
- What triggers a model retrain? How do you automate it?
- How do you handle rollback if a new model performs worse in production?
- Blue-green deployment vs canary deployment for ML models — tradeoffs?

**Production ML Monitoring**
- Your model is in production. What metrics do you monitor?
  - Model metrics: accuracy, F1, precision, recall on live data
  - System metrics: latency (p50, p95, p99), throughput, GPU/CPU/memory utilization
  - Data metrics: input distribution drift, feature drift, missing values
- What is data drift? How do you detect it? (PSI, KL divergence, KS test)
- What is concept drift vs data drift?
- How would you set up alerts for model degradation?
- Explain model serving options: FastAPI, TorchServe, vLLM, TGI — when would you pick each?

**Agentic Systems — from Infra Lens**
- How would you deploy an agentic system that makes multiple LLM calls per request?
- How do you handle cost control? Rate limiting? Budget caps per request?
- How would you monitor an agent's behavior in production? What logs do you capture?
- If an agent goes into an infinite loop making API calls, how do you detect and stop it?
- How would you set up the infrastructure for a multi-model serving system (e.g., different models for different agent capabilities)?

**Bash & Scripting**
- Write a bash script to: find all `.log` files older than 7 days and delete them
- Parse a CSV of model metrics and extract rows where accuracy < 0.9
- Write a script to health-check an ML API endpoint every 5 minutes
- How would you automate downloading, preprocessing, and uploading training data?

**SQL** (JD requirement)
- Write a query to find the top 5 models by accuracy from a model_registry table
- Window functions: running average of daily predictions
- JOIN model_predictions with ground_truth to compute accuracy per model per day

### How to Prepare

- Write down 3-4 specific examples of infra work you've done:
  - Multi-node GPU training at Krutrim (DeepSpeed, how many nodes, what went wrong)
  - How you deployed the healthcare reasoning model at Fractal
  - Any Docker/CI-CD experience you have
- Practice writing a Dockerfile from memory for a FastAPI + PyTorch app
- Review basic Linux commands: `ps`, `top`, `htop`, `df`, `du`, `lsof`, `netstat`, `grep`, `awk`, `sed`, `find`
- Review SQL JOINs, GROUP BY, HAVING, window functions (ROW_NUMBER, LAG, LEAD)
- Have a story ready about a production incident and how you debugged it

---

## Round 3: Satyandhra — Product / Strategy / Behavioral

### What to Expect

SDE background (4 years at Tower) + CMU MBA = he thinks in both engineering and business terms. He'll evaluate whether you can:
1. Think about *why* you're building something, not just *how*
2. Prioritize across competing demands
3. Communicate clearly to non-technical stakeholders
4. Understand the business context (trading, research acceleration)
5. Make sound cost/benefit tradeoffs

Since he's interviewing from the US at night, expect a more conversational, discussion-driven format rather than a coding test.

### Predicted Questions

**Agentic Systems — Product/Strategy Lens**
- You're building an autonomous post-training agent at Fractal. How did you decide what to build first?
- How do you measure success for an agent system? What does "good" look like to the business?
- How do you balance agent autonomy vs human oversight? Where do you put humans in the loop?
- If a researcher tells you "the agent isn't working well" — how do you debug that and prioritize fixes?
- How do you handle the cost-quality tradeoff? Cheaper models vs better results?
- Walk me through how you'd design an agent system for Tower's researchers — what would it do, how would you scope it?

**System Design (Product-Flavored)**
- Design a system where Tower's ML researchers can submit experiments, track results, and compare models. What components do you need?
- How would you design a data pipeline that ingests market data, processes it, and serves features to multiple ML models in real-time?
- Design an eval framework for an agentic system — how do you know if version 2 is better than version 1?
- How would you design a cost monitoring dashboard for LLM-powered agent systems?

**Prioritization & Decision-Making**
- You have 3 competing priorities: improving model accuracy, reducing inference latency, and building a new feature the researchers want. How do you decide what to work on?
- You shipped a model that performs well on benchmarks but researchers aren't using it. What do you do?
- How do you decide when to build vs buy? (e.g., build your own eval framework vs use an existing one)
- Tell me about a time you had to push back on a requirement. What happened?

**Communication & Stakeholder Management**
- How would you explain your tokenizer work to a non-technical person?
- You need to convince your manager to invest in building an agent system. How do you make the case?
- Tell me about a time you worked with people from different disciplines. How did you handle disagreements?

**Behavioral / Culture Fit**
- Why Tower? Why this role specifically?
- What's the most technically challenging problem you've solved?
- Tell me about a project that failed or didn't go as planned. What did you learn?
- Where do you see yourself in 3 years?
- How do you stay current with ML research? What's a recent paper that excited you?
- What's the difference between research and production ML? How do you bridge that gap?

**Finance / Trading Context**
- How familiar are you with financial time-series data? What's different about it compared to other domains?
- What challenges would you expect when deploying ML models in a trading context? (latency, regime changes, non-stationarity, look-ahead bias)
- How would you design an eval for a trading signal model vs a standard classification model?

### How to Prepare

- Prepare a **"pitch" for your agentic system** (2 min): problem → who benefits → architecture → results → what's next. Frame it in business terms, not just technical terms.
- For every resume item, know the **"so what"**: 500B tokens sounds impressive, but what business outcome did it drive? The tokenizer improved fertility — what did that mean in dollars/time saved?
- Prepare 3 behavioral stories using STAR format (Situation, Task, Action, Result):
  - A technical challenge you overcame
  - A time you influenced a decision
  - A project that didn't go as planned
- Think about **why Tower**: you have quant experience (Graviton), you build ML systems for production, you bridge research and engineering. Tower's Core AI/ML team needs exactly this.
- Have 2-3 good questions to ask *him* (e.g., "How does the Core AI/ML team's work get adopted by trading teams?" or "What does success look like for this role in the first 6 months?")

---

## 2-Day Prep Plan (Maximizing ROI)

### Day 1 — Foundation Day (heavier day)

**Morning (3-4 hours): ML/DL Theory + Resume Stories**

| Time | Activity | Why |
|------|----------|-----|
| 0:00-0:45 | Re-read prep doc sections 1-7 (Dijkstra/XOR, tokenization, PCA, norms/dropout/init, DeepSpeed, Python internals, bias-variance/MLE-MAP). Don't memorize — understand the *why* behind each concept | Covers Sarvagya's likely questions |
| 0:45-1:30 | Write down your **Fractal agent architecture** on paper. Draw the diagram. Note: what models, what tools the agent uses, how eval works, how cost is controlled, what breaks | This is the star of your interview — R1 feedback was specifically about this |
| 1:30-2:15 | Write down your **Krutrim stories**: tokenizer pipeline, synthetic data pipeline, Krutrim-2 post-training. For each: what was the problem, what did you do, what was the result, what would you do differently | Second most likely deep-dive topic |
| 2:15-3:00 | Review: attention mechanism math, KV-cache, RoPE, flash attention. Review model compression basics (pruning, quantization, distillation, LoRA/QLoRA) — these are Sarvagya's interest areas | |
| 3:00-3:30 | Rehearse your **Graviton story**: features used, model type, how you got 100x latency reduction | Directly relevant to Tower's trading context |

**Afternoon (3-4 hours): Infra/Systems + SQL**

| Time | Activity | Why |
|------|----------|-----|
| 0:00-0:45 | Write a Dockerfile from scratch for: `FastAPI + PyTorch model + GPU support`. Practice docker commands: build, run, push, volume mount, multi-stage | Neeraj will almost certainly ask Docker questions |
| 0:45-1:30 | Linux commands practice: `ps aux`, `top`, `kill`, `grep`, `find`, `awk`, `sed`, `chmod`, `cron`, `systemd`, `nvidia-smi`, `lsof`, `netstat`. Write a bash script that monitors GPU usage and alerts if >90% | |
| 1:30-2:15 | CI/CD mental model: draw the pipeline (git push → tests → train → evaluate → containerize → deploy → monitor → retrain). Know each step for ML specifically | |
| 2:15-3:00 | SQL review: write queries for JOINs, GROUP BY, HAVING, window functions (ROW_NUMBER, running averages with LAG/LEAD), subqueries. Practice on model_registry / predictions type tables | |
| 3:00-3:30 | Write down 2-3 infra war stories from your experience: multi-node training failures, deployment issues, debugging production problems | Stories > theory for Neeraj's round |

**Evening (1-2 hours): Product/Behavioral Prep**

| Time | Activity | Why |
|------|----------|-----|
| 0:00-0:30 | Write your **"why Tower"** answer. Connect: Graviton quant experience + Krutrim production ML + Fractal agentic systems = exactly what Tower's Core AI/ML needs | |
| 0:30-1:00 | Prepare 3 STAR stories: (1) hardest technical problem solved, (2) time you influenced a decision/pushed back, (3) something that failed and what you learned | Satyandhra will ask behavioral questions |
| 1:00-1:30 | Re-frame every resume bullet in business terms. "500B tokens" → what did it enable? "39.5% fertility improvement" → how much inference cost did it save? "100x latency reduction" → what was the production impact? | Product person cares about outcomes, not just methods |

### Day 2 — Sharpening Day (interview day or day before)

**Morning (2-3 hours): Coding + Practice**

| Time | Activity | Why |
|------|----------|-----|
| 0:00-1:00 | Do 2-3 LeetCode mediums: one DP (stock buy/sell variant), one sliding window, one graph. In Python, focus on clean code and talking through your approach | DSA might appear in any round (Sarvagya's R1 had DSA) |
| 1:00-1:30 | Code PCA from scratch without looking at the doc. Then code a simple neural network training loop (forward, loss, backward, step) from memory. These are "can you code ML" signals | |
| 1:30-2:00 | Practice SQL: "given tables `models(id, name, version)` and `evaluations(model_id, metric, value, date)`, find the best model per metric over the last 30 days" | |

**Afternoon (2-3 hours): Simulation + Weak Spots**

| Time | Activity | Why |
|------|----------|-----|
| 0:00-0:45 | **Mock Round 1**: Have a friend ask you ML questions OR talk through answers out loud. Explain PCA, explain why layer norm, explain DeepSpeed ZeRO stages — practice articulating clearly | Speaking practice is highest-ROI last-minute prep |
| 0:45-1:30 | **Mock Round 2**: Talk through how you'd deploy your Fractal agent system. Draw the infra: load balancer → API gateway → agent orchestrator → LLM calls → monitoring → logging. What metrics? What alerts? | |
| 1:30-2:15 | **Mock Round 3**: Practice your 2-min pitch for the agent system in business terms. Practice answering "how do you prioritize?" and "tell me about a failure" | |
| 2:15-2:45 | Review any weak spots you identified during the day. Fill gaps | |
| 2:45-3:00 | Prepare 2-3 questions for EACH interviewer. Smart questions signal genuine interest | |

---

## Quick-Reference: Key Concepts to Have Cold

### For Sarvagya (ML)
- Transformer attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- KV-cache: cache K,V from previous tokens during autoregressive generation
- RoPE: rotary position embedding, applies rotation matrix to Q,K based on position
- Flash attention: tiling attention computation to reduce memory from O(n²) to O(n)
- LoRA: low-rank adaptation, inject trainable rank-r matrices into frozen weights
- QLoRA: LoRA + 4-bit quantized base model
- RLHF pipeline: SFT → reward model → PPO (or DPO as simpler alternative)
- DPO: Direct Preference Optimization, skips reward model, directly optimizes policy

### For Neeraj (Infra)
- Dockerfile template:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- Key monitoring stack: Prometheus (metrics) + Grafana (dashboards) + ELK (logs)
- Model serving options: vLLM (LLM inference), TorchServe (general PyTorch), FastAPI (lightweight)
- GPU debugging: `nvidia-smi`, `torch.cuda.memory_summary()`, OOM = reduce batch size or use gradient checkpointing

### For Satyandhra (Product)
- Frame every answer as: **Problem → Approach → Outcome → Learnings**
- When asked "how would you design X" → start with **user needs**, not architecture
- Cost of LLM calls: GPT-4 class ~$10-30/M tokens, smaller models ~$0.1-1/M tokens
- Evals for agents: task completion rate, cost per task, latency, human satisfaction, error rate
- Why Tower: "I've built ML systems for production at Krutrim and agentic systems at Fractal. I have quant context from Graviton. Tower's Core AI/ML team sits at the intersection of all three — building production ML infrastructure that accelerates quantitative research."

---

## Anti-Patterns to Avoid

- **Don't just describe what you did — explain why.** "I used DeepSpeed ZeRO-3" → "We used ZeRO-3 because our 7B model's optimizer states alone exceeded single-GPU memory, and ZeRO-3 let us partition everything across 8 GPUs"
- **Don't answer infra questions with "I just used X."** Neeraj wants to know you understand what's happening under the hood
- **Don't be purely technical with Satyandhra.** He has an MBA — he thinks in terms of impact, cost, users, and priorities
- **Don't wing the behavioral questions.** Prepare specific stories. Vague answers signal you haven't reflected on your experience
- **If you don't know something, say so and reason through it.** "I haven't worked with X directly, but based on my understanding of Y, I'd approach it like..."
