# 🚌 Pareto Transport Scheduler

### 📘 Project Overview  
The **Pareto Transport Scheduler** is a **Streamlit-based interactive web application** designed to optimize public transport scheduling using **multi-objective evolutionary algorithms** — specifically the **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**.

The goal is to balance two conflicting objectives:
- **Operational Cost Minimization** 💰  
- **Passenger Waiting Time Reduction** ⏱️  

By applying **Pareto-based multi-objective optimization**, the system identifies a set of optimal trade-off solutions, helping planners make data-driven scheduling decisions.

---


### ⚙️ Features
✅ **Interactive UI** built with Streamlit and Plotly  
✅ **NSGA-II Algorithm Implementation** for multi-objective optimization  
✅ **Pareto Front Visualization** showing the cost–waiting time trade-off  
✅ **Customizable Parameters** via sidebar sliders (population, generations, bus limits)  
✅ **Dynamic Metrics Dashboard** displaying efficiency, satisfaction, and cost  
✅ **Modern UI Styling** with gradient backgrounds, animations, and metric cards  

---

### 🧮 Optimization Objectives
- **Objective 1: Minimize Total Operational Cost (₹)**  
  Computed based on bus count, route length, and cost per km.  
- **Objective 2: Minimize Average Passenger Waiting Time (min)**  
  Estimated using bus frequency on each route.

The algorithm identifies **Pareto-optimal solutions** — where improving one objective cannot be done without worsening the other.

---

### 🧠 Algorithm Used: NSGA-II (Non-Dominated Sorting Genetic Algorithm II)
The project implements **NSGA-II**, a popular evolutionary algorithm for multi-objective optimization.  
Key steps include:
1. **Population Initialization** – Random generation of possible bus allocations.  
2. **Evaluation** – Calculating cost and waiting time for each individual.  
3. **Fast Non-Dominated Sorting** – Grouping individuals into Pareto fronts.  
4. **Crowding Distance Assignment** – Preserving diversity among solutions.  
5. **Selection, Crossover, and Mutation** – Generating next generations.  
6. **Termination** – Best Pareto front displayed after all generations.

---

### 🧰 Tech Stack

| **Component** | **Technology** |
|----------------|----------------|
| Frontend/UI | Streamlit |
| Backend/Logic | Python |
| Data Handling | NumPy, Pandas |
| Visualization | Plotly Express, Plotly Graph Objects |
| Optimization Algorithm | NSGA-II (implemented manually) |

---

### 🖥️ How to Run the Project

1. Save the project code as **`pareto_transport_scheduler.py`**  
2. Run the Streamlit app using the command:
   ```bash
   streamlit run pareto_transport_scheduler.py
   ```
The app will launch in your browser (default: http://localhost:8501)

### 📊 Output Screens
Home Screen: Introduction and instructions

Pareto Front Plot: Interactive scatter plot of cost vs. waiting time

Metrics Dashboard: Displays best recommended solution with frequency, cost, satisfaction, and efficiency

Summary Section: Highlights Pareto count, cost range, and waiting time range

### 🎯 Results & Insights

* The system finds multiple Pareto-optimal solutions balancing cost and service quality.
* The recommended solution offers the lowest average waiting time while keeping costs manageable.
* Efficiency Score indicates the quality points per ₹1,000 spent — providing actionable insights for planners.

### 👩‍💻 Team Members

* Akshada Mane
* Shubhangi Nimbalkar
* Sakshi Hedke
* Isha Haval

### 🏆 Acknowledgement
Developed as part of a **Soft Computing** course SCE under the guidance of **Dr. Laxmi Bewoor Ma’am** at **Vishwakarma Institute of Information Technology, Pune**, focusing on **Pareto-Based Multi-Objective Optimization for Public Transport Scheduling.**
