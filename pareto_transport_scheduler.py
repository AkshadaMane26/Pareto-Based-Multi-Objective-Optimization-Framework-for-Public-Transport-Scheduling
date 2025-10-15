import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

# ------------------------- Page Setup -------------------------
st.set_page_config(page_title="Pareto Transport Scheduler", page_icon="🚌", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {font-family: 'Inter', sans-serif;}
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        box-shadow: 4px 0 20px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stSidebar"] label {
        color: white !important; 
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    div[data-testid="stSidebar"] h2 {
        color: white !important;
        font-size: 20px;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        margin-bottom: 20px;
    }
    
    /* Slider styling */
    div[data-testid="stSidebar"] .stSlider > div > div {
        background: rgba(255,255,255,0.2);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #1f2937;
        font-weight: 700;
        font-size: 16px;
        padding: 15px 30px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(67,233,123,0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(67,233,123,0.6);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 700;
    }
    
    /* Info box styling */
    .stAlert {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 20px;
        font-size: 16px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div style='text-align: center; padding: 40px 20px; background: white; border-radius: 20px; 
     box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 40px; position: relative; overflow: hidden;'>
    <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
         background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);'></div>
    <div style='position: relative; z-index: 1;'>
        <h1 style='margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
             font-size: 48px; font-weight: 800; letter-spacing: -1px;'>
            🚌 Pareto Transport Scheduler
        </h1>
        <p style='color: #6b7280; font-size: 20px; margin-top: 15px; font-weight: 500;'>
            Optimize public transport scheduling using multi-objective evolutionary algorithms
        </p>
        <div style='margin-top: 20px; display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 24px;'>⚡</span>
                <span style='color: #667eea; font-weight: 600;'>NSGA-II Algorithm</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 24px;'>📊</span>
                <span style='color: #667eea; font-weight: 600;'>Pareto Optimization</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 24px;'>🎯</span>
                <span style='color: #667eea; font-weight: 600;'>Cost-Quality Balance</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------- Utility Functions -------------------------
def create_synthetic_routes(n_routes: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    routes = []
    for i in range(n_routes):
        length_km = round(float(rng.uniform(5.0, 40.0)), 1)
        cost_per_km = round(float(rng.uniform(1.0, 2.0)), 2)
        routes.append((f"R{i+1}", length_km, cost_per_km))
    return pd.DataFrame(routes, columns=["route_id", "length_km", "cost_per_km"])

def evaluate_population(pop: np.ndarray, routes: pd.DataFrame, min_buses: int, max_buses: int) -> List[Tuple[float, float]]:
    results = []
    lengths = routes["length_km"].values
    cost_per_km = routes["cost_per_km"].values
    k = 10
    for ind in pop:
        buses = np.clip(np.rint(ind), min_buses, max_buses)
        # Ensure no zeros
        buses = np.where(buses == 0, 1, buses)
        waiting_time = np.mean(k / buses)
        cost = (buses * lengths * cost_per_km).sum()
        results.append((float(cost), float(waiting_time)))
    return results

def fast_non_dominated_sort(values: List[Tuple[float, float]]):
    S = [set() for _ in range(len(values))]
    n = [0]*len(values)
    fronts = [[]]
    for p in range(len(values)):
        for q in range(len(values)):
            if p==q: continue
            if (values[p][0]<=values[q][0] and values[p][1]<=values[q][1]) and (values[p][0]<values[q][0] or values[p][1]<values[q][1]):
                S[p].add(q)
            elif (values[q][0]<=values[p][0] and values[q][1]<=values[p][1]) and (values[q][0]<values[p][0] or values[q][1]<values[p][1]):
                n[p] +=1
        if n[p]==0: fronts[0].append(p)
    i=0
    while fronts[i]:
        next_front=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0: next_front.append(q)
        i+=1
        fronts.append(next_front)
    fronts.pop()
    return fronts

def crowding_distance(values: List[Tuple[float, float]], front: List[int]):
    distances = {i:0.0 for i in front}
    if not front: return distances
    for m in range(2):
        front_sorted = sorted(front, key=lambda idx: values[idx][m])
        distances[front_sorted[0]] = distances[front_sorted[-1]] = float('inf')
        obj_min = values[front_sorted[0]][m]
        obj_max = values[front_sorted[-1]][m]
        if obj_max - obj_min == 0: continue
        for i in range(1,len(front_sorted)-1):
            prev_val = values[front_sorted[i-1]][m]
            next_val = values[front_sorted[i+1]][m]
            distances[front_sorted[i]] += (next_val-prev_val)/(obj_max-obj_min)
    return distances

def simulated_binary_crossover(parents: np.ndarray, eta: float=15.0, lb: float=0.0, ub: float=20.0):
    pop_size, n_var = parents.shape
    offspring = np.empty_like(parents)
    for i in range(0,pop_size,2):
        parent1 = parents[i]
        parent2 = parents[i+1 if i+1<pop_size else 0]
        for j in range(n_var):
            if np.random.rand()<=0.9 and abs(parent1[j]-parent2[j])>1e-14:
                x1,x2 = min(parent1[j],parent2[j]), max(parent1[j],parent2[j])
                rand = np.random.rand()
                beta = 1 + 2*(x1-lb)/(x2-x1)
                alpha = 2-beta**-(eta+1)
                betaq = (rand*alpha)**(1/(eta+1)) if rand<=1/alpha else (1/(2-rand*alpha))**(1/(eta+1))
                c1 = 0.5*((x1+x2)-betaq*(x2-x1))
                offspring[i,j] = np.clip(c1,lb,ub)
            else:
                offspring[i,j] = parent1[j]
    return offspring

def polynomial_mutation(offspring: np.ndarray, eta: float=20.0, mut_prob: float=0.1, lb: float=0.0, ub: float=20.0):
    pop_size, n_var = offspring.shape
    for i in range(pop_size):
        for j in range(n_var):
            if np.random.rand()<mut_prob:
                x = offspring[i,j]
                delta1 = (x-lb)/(ub-lb)
                delta2 = (ub-x)/(ub-lb)
                rand = np.random.rand()
                mut_pow = 1/(eta+1)
                try:
                    if rand<0.5:
                        xy=1-delta1
                        val=2*rand+(1-2*rand)*(xy**(eta+1))
                        deltaq=val**mut_pow-1
                    else:
                        xy=1-delta2
                        val=2*(1-rand)+2*(rand-0.5)*(xy**(eta+1))
                        deltaq=1-val**mut_pow
                except OverflowError:
                    deltaq = 0
                x=x+deltaq*(ub-lb)
                offspring[i,j]=np.clip(x,lb,ub)
    return offspring

# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.markdown("<h2>⚙️ Algorithm Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("### Population")
    pop_size = st.slider("Population Size", 20, 200, 80, help="Number of solutions in each generation")
    
    st.markdown("### Evolution")
    n_generations = st.slider("Generations", 5, 200, 50, help="Number of evolutionary iterations")
    
    st.markdown("### Bus Constraints")
    min_buses = st.slider("Minimum Buses/Hour", 1, 10, 2)
    max_buses = st.slider("Maximum Buses/Hour", 10, 50, 20)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🚀 Run Optimization")
    

routes_df = create_synthetic_routes(6)

# ------------------------- Execution -------------------------
if run_button:
    np.random.seed(42)
    n_var = len(routes_df)
    pop = np.random.uniform(min_buses,max_buses,size=(pop_size,n_var))
    
    # Progress container
    progress_container = st.container()
    with progress_container:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 16px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); margin-bottom: 30px;'>
            <h3 style='margin: 0 0 15px 0; color: #667eea;'>🔄 Optimization in Progress</h3>
        </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    all_costs=[]
    all_waiting=[]

    for gen in range(n_generations):
        values = evaluate_population(pop,routes_df,min_buses,max_buses)
        offspring = simulated_binary_crossover(pop, lb=min_buses, ub=max_buses)
        offspring = polynomial_mutation(offspring, lb=min_buses, ub=max_buses)
        union = np.vstack((pop,offspring))
        union_values = evaluate_population(union,routes_df,min_buses,max_buses)
        fronts = fast_non_dominated_sort(union_values)

        new_pop=[]
        for front in fronts:
            if len(new_pop)+len(front)>pop_size:
                distances = crowding_distance(union_values,front)
                sorted_front=sorted(front,key=lambda idx: distances[idx],reverse=True)
                for idx in sorted_front[:pop_size-len(new_pop)]:
                    new_pop.append(union[idx])
                break
            else:
                for idx in front:
                    new_pop.append(union[idx])
        pop=np.array(new_pop)

        for val in union_values:
            all_costs.append(val[0])
            all_waiting.append(val[1])
        
        progress_bar.progress(int((gen+1)/n_generations*100))
        status_text.text(f"Generation {gen+1}/{n_generations} - Evaluating {len(fronts[0])} Pareto-optimal solutions")

    status_text.text("✅ Optimization Complete!")

    final_values = evaluate_population(pop,routes_df,min_buses,max_buses)
    fronts = fast_non_dominated_sort(final_values)
    pareto_idx = fronts[0]

    costs = [final_values[i][0] for i in pareto_idx]
    waiting_times = [final_values[i][1] for i in pareto_idx]

    # Recommended solution
    # Recommended solution - safe version
    best_idx = pareto_idx[np.argmin(waiting_times)]

    # Safely compute values
    best_buses = np.clip(np.rint(pop[best_idx]), min_buses, max_buses)
    avg_frequency = float(np.nan_to_num(np.mean(best_buses), nan=min_buses))
    cost = float(np.nan_to_num(final_values[best_idx][0], nan=0.0))
    wait_time = float(np.nan_to_num(final_values[best_idx][1], nan=1.0))  # avoid divide by zero
    service_quality = 10 / wait_time if wait_time > 0 else 0
    satisfaction = max(0, 100 - wait_time*5)
    efficiency = service_quality / (cost / 1000) if cost > 0 else 0

    recommended = {
        'Frequency (min)': avg_frequency,
        'Cost ($)': cost,
        'Service Quality': service_quality,
        'Satisfaction (%)': satisfaction,
        'Efficiency': efficiency
    }


    # ------------------------- Metrics Cards -------------------------
    st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 40px;'>
            <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 20px;'>
                <span style='font-size: 36px;'>🎯</span>
                <h2 style='margin: 0; color: #667eea; font-size: 32px;'>Recommended Solution</h2>
            </div>
            <p style='color: #6b7280; font-size: 16px; margin: 0;'>
                This solution provides the best service quality while maintaining reasonable operational costs
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #667eea, #764ba2); color: white;'>
                <div style='font-size: 40px; margin-bottom: 10px;'>🚌</div>
                <div style='font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Frequency</div>
                <div style='font-size: 36px; font-weight: 800;'>{int(np.nan_to_num(recommended['Frequency (min)'], nan=min_buses))}</div>
                <div style='font-size: 14px; opacity: 0.8; margin-top: 5px;'>minutes</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #f093fb, #f5576c); color: white;'>
                <div style='font-size: 40px; margin-bottom: 10px;'>💰</div>
                <div style='font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Daily Cost</div>
                <div style='font-size: 36px; font-weight: 800;'>₹{recommended['Cost ($)']:.0f}</div>
                <div style='font-size: 14px; opacity: 0.8; margin-top: 5px;'>operational</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #4facfe, #00f2fe); color: white;'>
                <div style='font-size: 40px; margin-bottom: 10px;'>📈</div>
                <div style='font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Service Quality</div>
                <div style='font-size: 36px; font-weight: 800;'>{recommended['Service Quality']:.1f}</div>
                <div style='font-size: 14px; opacity: 0.8; margin-top: 5px;'>score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #43e97b, #38f9d7); color: white;'>
                <div style='font-size: 40px; margin-bottom: 10px;'>😊</div>
                <div style='font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Satisfaction</div>
                <div style='font-size: 36px; font-weight: 800;'>{recommended['Satisfaction (%)']:.0f}%</div>
                <div style='font-size: 14px; opacity: 0.8; margin-top: 5px;'>estimated</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ffecd2, #fcb69f); padding: 25px; border-radius: 16px; 
             margin: 30px 0; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
            <div style='display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;'>
                <span style='font-size: 32px;'>⚡</span>
                <div>
                    <div style='font-size: 24px; font-weight: 800; color: #1f2937;'>
                        Efficiency Score: {recommended['Efficiency']:.2f}
                    </div>
                    <div style='font-size: 14px; color: #6b7280; margin-top: 5px;'>
                        Quality points per ₹1,000 spent
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------------- Pareto Front -------------------------
    st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin: 40px 0 20px 0;'>
            <h2 style='margin: 0; color: #1f2937;'>📊 Pareto Front Analysis</h2>
            <p style='color: #6b7280; margin-top: 10px;'>
                Explore the trade-off between operational costs and waiting times. Each point represents an optimal solution.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    size = []
    for i in pareto_idx:
        freq = np.mean(np.clip(np.rint(pop[i]), min_buses, max_buses))
        if np.isnan(freq) or np.isinf(freq):
            freq = min_buses
        size.append(freq)

    fig = px.scatter(
        x=costs,
        y=waiting_times,
        labels={"x": "Operational Cost (₹)", "y": "Avg Waiting Time (min)"},
        title="",
        size=size,
        color=waiting_times,
        color_continuous_scale="Viridis"
    )
    
    fig.update_traces(marker=dict(line=dict(width=2, color='white')))
    fig.update_layout(
        xaxis=dict(tickformat=",d", tickprefix="₹", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        template="plotly_white",
        height=500,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary Statistics
    st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin: 40px 0 20px 0;'>
            <h2 style='margin: 0; color: #1f2937;'>📋 Optimization Summary</h2>
        </div>
    """, unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); 
                 padding: 25px; border-radius: 16px; text-align: center; color: #1f2937;'>
                <div style='font-size: 16px; font-weight: 600; margin-bottom: 10px;'>
                    Pareto Solutions Found
                </div>
                <div style='font-size: 42px; font-weight: 800;'>
                    {len(pareto_idx)}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%); 
                 padding: 25px; border-radius: 16px; text-align: center; color: #1f2937;'>
                <div style='font-size: 16px; font-weight: 600; margin-bottom: 10px;'>
                    Cost Range
                </div>
                <div style='font-size: 42px; font-weight: 800;'>
                    ₹{min(costs):.0f} - ₹{max(costs):.0f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #fdcbf1 0%, #e6dee9 100%); 
                 padding: 25px; border-radius: 16px; text-align: center; color: #1f2937;'>
                <div style='font-size: 16px; font-weight: 600; margin-bottom: 10px;'>
                    Wait Time Range
                </div>
                <div style='font-size: 42px; font-weight: 800;'>
                    {min(waiting_times):.1f} - {max(waiting_times):.1f} min
                </div>
            </div>
        """, unsafe_allow_html=True)
    

else:
    # Welcome screen when not running
    st.markdown("""
        <div style='background: white; padding: 50px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); 
             text-align: center; margin: 40px 0;'>
            <div style='font-size: 80px; margin-bottom: 20px;'>🚀</div>
            <h2 style='color: #1f2937; margin-bottom: 15px;'>Ready to Optimize?</h2>
            <p style='color: #6b7280; font-size: 18px; max-width: 600px; margin: 0 auto 30px auto; line-height: 1.8;'>
                Configure the algorithm parameters in the sidebar and click 
                <strong style='color: #667eea;'>"Run Optimization"</strong> to find the best 
                balance between cost and service quality for your transport network.
            </p>
            <div style='display: flex; justify-content: center; gap: 40px; margin-top: 40px; flex-wrap: wrap;'>
                <div style='text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>⚙️</div>
                    <div style='font-weight: 600; color: #1f2937; margin-bottom: 5px;'>Configure</div>
                    <div style='color: #6b7280; font-size: 14px;'>Set parameters</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>🔄</div>
                    <div style='font-weight: 600; color: #1f2937; margin-bottom: 5px;'>Optimize</div>
                    <div style='color: #6b7280; font-size: 14px;'>Run NSGA-II</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>📊</div>
                    <div style='font-weight: 600; color: #1f2937; margin-bottom: 5px;'>Analyze</div>
                    <div style='color: #6b7280; font-size: 14px;'>View results</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("""
        <div style='background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin: 40px 0;'>
            <h2 style='text-align: center; color: #1f2937; margin-bottom: 30px;'>✨ Key Features</h2>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px;'>
                <div style='text-align: center;'>
                    <div style='font-size: 40px; margin-bottom: 15px;'>🎯</div>
                    <h3 style='color: #667eea; font-size: 18px; margin-bottom: 10px;'>Multi-Objective</h3>
                    <p style='color: #6b7280; font-size: 14px; line-height: 1.6;'>
                        Simultaneously optimize cost and service quality using advanced evolutionary algorithms
                    </p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 40px; margin-bottom: 15px;'>📈</div>
                    <h3 style='color: #667eea; font-size: 18px; margin-bottom: 10px;'>Pareto Analysis</h3>
                    <p style='color: #6b7280; font-size: 14px; line-height: 1.6;'>
                        Discover optimal trade-offs with non-dominated sorting and crowding distance
                    </p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 40px; margin-bottom: 15px;'>📊</div>
                    <h3 style='color: #667eea; font-size: 18px; margin-bottom: 10px;'>Visual Insights</h3>
                    <p style='color: #6b7280; font-size: 14px; line-height: 1.6;'>
                        Interactive charts and dashboards for comprehensive result analysis
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
    st.markdown("""
        <div style='text-align: center; padding: 30px; margin-top: 60px; color: #6b7280; font-size: 14px;'>
            <div style='margin-bottom: 10px;'>
                <strong style='color: #667eea;'>Pareto Transport Scheduler</strong> | 
                Powered by NSGA-II Algorithm
            </div>
            <div>
                Multi-objective optimization for smart transport planning
            </div>
        </div>
    """, unsafe_allow_html=True)
