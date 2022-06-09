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

def max(log_file):
    max_cpu, max_mem, max_out = [0,0,0]

    for task in log_file:
        task = task.strip().split(',')
        _, category, status, _, _, _, _, _, _, _, cpu_time, memory, _, fout = task

        if (status == 'accumulated') and (not category == 'preprocessing'):
            max_cpu  = float(cpu_time) if float(cpu_time) > max_cpu else max_cpu
            max_mem  = float(memory) if float(memory) > max_mem else max_mem
            max_out  = float(fout) if float(fout) > max_out else max_out

    return max_cpu, max_mem, max_out

def parse_log_file(log_file, workflow, max_cpu, max_mem, max_out):
    for task in log_file:
        task = task.strip().split(',')
        task_id, category, status, _, _, range_start, range_stop, accum_parent, time_start, time_end, cpu_time, memory, _, fout = task

        # Create nodes and directed edges between processing and accumulation tasks
        # Tasks must have a status of "accumulated"
        # Ignore preprocessing tasks
        if (status == 'accumulated') and (not category == 'preprocessing'):
            func_size  = math.sqrt(float(memory)/max_mem)
            func_color = float(cpu_time)/max_cpu
            fout_size  = float(fout)/max_out

            # Function node
            func_node_info = f'''ID: {task_id}{category[0]}
CPU: {round(float(cpu_time), 3)}s
Wall: {round(float(time_end) - float(time_start), 3)}s
{memory} MB
{float(range_stop) - float(range_start)}'''

            workflow.node(task_id, label=func_node_info,
                    width       = f'{(func_size + 1)*30}',
                    height      = f'{(func_size + 1)*30}',
                    fontsize    = '300',
                    fontname    = 'Helvetica',
                    colorscheme = 'rdylgn11',
                    style       = 'filled',
                    color       = f'{11 - int(func_color*10)}')

            # Output node
            workflow.node(f'{task_id}{fout}', label=f'{size(float(fout))}B',
                    shape       = 'box',
                    width       = f'{(fout_size + 1)*30}',
                    height      = f'{(fout_size + 1)*30}',
                    fontsize    = '500',
                    fontname    = 'Helvetica')

            # Edge between function and output
            workflow.edge(task_id, f'{task_id}{fout}')

            # Edge between output and accumulation
            if not accum_parent == '0':
                workflow.edge(f'{task_id}{fout}', accum_parent)

    return workflow

def generate_dot(workflow):
    workflow_file = open(f'{sys.argv[1]}.wf.gv', 'w')
    workflow_file.writelines(str(workflow))
    workflow_file.close()

def generate_viz(log_file):
    os.system(f'dot -Tpdf {log_file}.wf.gv -o {log_file}.wf.pdf')

# Main execution

def main():
    # Open log file
    log_file = open(sys.argv[1], 'r')

    # Determine max CPU time, memory, and output size usage for coloring/scaling
    max_cpu, max_mem, max_out = max(log_file)
    log_file.seek(0) # Reset file handle to beginning of file

    # Parse log file and generate DOT object
    workflow = gv.Digraph('TopEFT Workflow', graph_attr={'rankdir': 'LR', 'size': '85,110', 'ratio': 'fill'})
    workflow = parse_log_file(log_file, workflow, max_cpu, max_mem, max_out)

    # Generate DOT (.gv) and visualization (.pdf) files
    # .gv files contain DOT syntax to organize the graph
    generate_dot(workflow)
    generate_viz(sys.argv[1])

    log_file.close()

if __name__ == '__main__':
    main()
