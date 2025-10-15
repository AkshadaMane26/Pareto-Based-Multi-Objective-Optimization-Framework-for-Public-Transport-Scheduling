# app.py - Single page Flask app
from flask import Flask, render_template, request, send_from_directory
from optimization import run_optimization
from pareto_plot import plot_pareto_front
import pandas as pd
import time

app = Flask(__name__, static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = None
    image_path = None
    params = {'population':50, 'generations':50, 'freq_min':1, 'freq_max':6}
    runtime = None

    if request.method == 'POST':
        try:
            params['population'] = int(request.form.get('population', 50))
            params['generations'] = int(request.form.get('generations', 50))
            params['freq_min'] = int(request.form.get('freq_min', 1))
            params['freq_max'] = int(request.form.get('freq_max', 6))
        except ValueError:
            pass

        # Validate bounds
        if params['freq_min'] < 1: params['freq_min'] = 1
        if params['freq_max'] < params['freq_min']: params['freq_max'] = params['freq_min']

        start = time.time()
        pareto_front = run_optimization(n_pop=params['population'], n_gen=params['generations'],
                                       freq_min=params['freq_min'], freq_max=params['freq_max'])
        runtime = round(time.time() - start, 3)
        image_path = plot_pareto_front(pareto_front, save_path='static/img/pareto.png')

        top_schedules = []
        for ind in pareto_front[:10]:
            top_schedules.append({
                'Route 1': ind[0],
                'Route 2': ind[1],
                'Route 3': ind[2],
                'Cost (₹)': round(ind.fitness.values[0],2),
                'Avg Wait (min)': round(ind.fitness.values[1],2)
            })

        df = pd.DataFrame(top_schedules)
        result_html = df.to_html(classes='table table-dark table-striped table-bordered', index=False, justify='center')

    return render_template('index.html', result=result_html, image=image_path, params=params, runtime=runtime)

# Serve favicon if present (optional)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
