# PublicTransportFlaskSinglePage
Pareto-Based Multi-Objective Optimization for Public Transport Scheduling
Single-page Flask app with interactive parameters and an attractive dark navy UI.

## How to run
1. Create a virtual environment (recommended):
   python -m venv venv
   On Windows: venv\Scripts\activate
   On macOS/Linux: source venv/bin/activate

2. Install requirements:
   pip install -r requirements.txt

3. Run the app:
   python app.py

4. Open your browser at http://127.0.0.1:5000/

## Files
- app.py: Main Flask app (single page)
- optimization.py: DEAP NSGA-II optimization (dynamic frequency range)
- objectives.py: Cost and waiting time calculations
- simulation.py: Route / cost parameters
- pareto_plot.py: Saves Pareto front plot to static/img/pareto.png
- templates/index.html: Single-page UI (dark navy theme)
- static/css/style.css: Custom styles and effects
