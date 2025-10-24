# LLM4AD Optimization Experiments

This repository provides experiments for running optimization algorithms with LLM guidance.  
Supported algorithms include **MOMCTS**, **MEoH**, and **NSGA2**, tested on problems like **Multi-objective TSP** and **Online Bin Packing**.  
The system uses the **Gemini API** as the LLM backend and stores results in structured logs for analysis.

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/llm4ad.git
   cd llm4ad
  
2. **Set up API Keys**
Create a .env file in the project root and add your Gemini API keys:
  ```bash
  API_KEY1=your_first_api_key
  API_KEY2=your_second_api_key
  # Add more as needed: API_KEY3, API_KEY4, ...
  ```

3. **Configuration**
You can configure the algorithm and problem inside the script.

Algorithms:
momcts
meoh
nsga2

Problems:
tsp (Multi-objective Traveling Salesman)
bpo (Online Bin Packing)

Example configuration:
```bash
ALGORITHM_NAME = 'momcts'   # Options: 'momcts', 'meoh', 'nsga2'
PROBLEM_NAME = "tsp"        # Options: "tsp", "bpo"