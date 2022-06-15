#!/usr/bin/env python3

# Author: Andrew Hennessee
# Date: 5/26/2022
# Program to parse topEFT task accumulation log file and produce visualization of task workflow

# Libraries

import os
import sys
import graphviz as gv
import math
from hurry.filesize import size

# Functions

def parse(log_file):
    tasks = []
    max_cpu = 0
    max_mem = 0
    max_in  = 0
    max_out = 0

    next(log_file) # Skip header
    for task in log_file:
        task = task.strip().split(',')
        task_info = { # See tasks_accum_log
            'task_id'      : task[0],
            'category'     : task[1],
            'status'       : task[2],
            'accum_parent' : task[7],
            'time_start'   : float(task[8]),
            'time_end'     : float(task[9]),
            'cpu_time'     : float(task[10]),
            'memory'       : float(task[11]),
            'fin'          : float(task[12]),
            'fout'         : float(task[13]),
        }

        # Track accumulated tasks, ignore preprocessing
        # Determine max CPU time, memory usage, and output size (for scaling)
        if (task_info['status'] == 'accumulated') and (task_info['category'] != 'preprocessing'):
            tasks.append(task_info)
            max_cpu = task_info['cpu_time'] if task_info['cpu_time'] > max_cpu else max_cpu
            max_mem = task_info['memory'] if task_info['memory'] > max_mem else max_mem
            max_out = task_info['fout'] if task_info['fout'] > max_out else max_out

    return tasks, max_cpu, max_mem, max_out

def make_graph(tasks, workflow, max_cpu, max_mem, max_out):
    # Create nodes and directed edges between processing and accumulation tasks
    for task in tasks:

        # Function node
        func_node_info = f'''ID: {task['task_id']}{task['category'][0]}
CPU: {round(task['cpu_time'], 3)}s
Wall: {round(task['time_end'] - task['time_start'], 3)}s
Mem: {task['memory']} MB
{f"In: {size(task['fin'])}" if task['fin'] != 0 else ""}'''

        func_size  = math.sqrt(task['memory']/max_mem)
        func_color = task['cpu_time']/max_cpu
        workflow.node(task['task_id'], label=func_node_info,
            width       = f'{(func_size + 1)*30}',
            height      = f'{(func_size + 1)*30}',
            fontsize    = '300',
            fontname    = 'Helvetica',
            colorscheme = 'rdylgn11',
            style       = 'filled',
            color       = f'{11 - int(func_color*10)}')

        # Output node
        fout_size = task['fout']/max_out
        workflow.node(f'''{task['task_id']}{task['fout']}''',
            label=f'''Out:\n{size(task['fout'])}\n{f"({round(task['fout']/task['fin']*100, 2)}%)" if task['fin'] != 0 else ""}''',
            shape       = 'box',
            width       = f'{(fout_size + 1)*30}',
            height      = f'{(fout_size + 1)*30}',
            fontsize    = '500',
            fontname    = 'Helvetica')

        # Edge between function and output
        workflow.edge(task['task_id'], f'''{task['task_id']}{task['fout']}''')

        # Edge between output and accumulation
        if not task['accum_parent'] == '0':
            workflow.edge(f'''{task['task_id']}{task['fout']}''', task['accum_parent'])

    return workflow

def generate_dot(workflow):
    workflow_file = open(f'{sys.argv[1]}.gv', 'w')
    workflow_file.writelines(str(workflow))
    workflow_file.close()

def generate_viz(log_file):
    os.system(f'dot -Tpdf {log_file}.gv -o {log_file}.pdf')

# Main execution

def main():
    # Parse log file to collect info for processing and accumulating tasks
    log_file = open(sys.argv[1], 'r')
    tasks, max_cpu, max_mem, max_out = parse(log_file)
    log_file.close()

    # Generate DOT object
    workflow = gv.Digraph('TopEFT Workflow', graph_attr={'rankdir': 'LR', 'size': '85,110', 'ratio': 'fill'})
    workflow = make_graph(tasks, workflow, max_cpu, max_mem, max_out)

    # Generate DOT (.gv) and visualization (.pdf) files
    # .gv files contain DOT syntax to organize the graph
    generate_dot(workflow)
    generate_viz(sys.argv[1])

if __name__ == '__main__':
    main()
