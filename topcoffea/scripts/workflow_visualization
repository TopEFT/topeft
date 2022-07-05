#!/usr/bin/env python3

# Author: Andrew Hennessee
# Date: 5/26/2022
# Program to parse topEFT task accumulation log file and produce visualization of task workflow

# Packages and definitions
import os
import sys
import math

# The graphviz package facilitates the creation and rendering of graph descriptions
# in the DOT language of the Graphviz graph drawing software from Python.
# Create a graph object, assemble the graph by adding nodes and edges,
# and retrieve its DOT source code string. Save the source code to a file and
# render it with the Graphviz installation of your system.
try:
    import graphviz as gv
except ImportError:
    print('graphviz is not installed. Please run the following command to install:')
    print('pip install graphviz')

# The hurry.filesize package is a simple Python library that can take a number of bytes and
# returns a human-readable string with the size in it, in kilobytes (K), megabytes
# (M), etc.
try:
    from hurry.filesize import size
except ImportError:
    print('hurry.filesize is not installed. Please run the following command to install:')
    print('pip install hurry.filesize')


# Functions
def usage(status=0, prog=os.path.basename(__file__)):
    print(f'Usage: {prog} [-o outfile] logfile')
    print('''
  -o  Specify the output file name, default name is <logfile>.pdf.
''')
    sys.exit(status)


def parse(logfile):
    tasks = []
    max_cpu = 0
    max_mem = 0
    max_in  = 0
    max_out = 0

    next(logfile) # Skip header
    for task in logfile:
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


def generate_dot(workflow, outfile):
    workflow_file = open(f'{outfile}.gv', 'w')
    workflow_file.writelines(str(workflow))
    workflow_file.close()


def generate_viz(outfile):
    os.system(f'dot -Tpdf {outfile}.gv -o {outfile}.pdf')


# Main execution
def main():
    arguments = sys.argv[1:]
    logfile   = ''
    outfile   = ''

    while arguments:
        arg = arguments.pop(0)
        if arg[0] == '-':
            if arg == '-o':
                outfile = arguments.pop(0)
            elif arg == '-h':
                usage()
            else:
                usage(1)
        else:
            logfile = arg

    if not logfile:
        print('No log file specified.')
        usage(1)

    if not outfile:
        outfile = logfile

    try:
        logfile = open(logfile, 'r')
    except FileNotFoundError:
        print(f'{logfile} not found.')
        usage(1)

    tasks, max_cpu, max_mem, max_out = parse(logfile)

    workflow = gv.Digraph('TopEFT Workflow', graph_attr={'rankdir': 'LR', 'size': '85,110', 'ratio': 'fill'})
    workflow = make_graph(tasks, workflow, max_cpu, max_mem, max_out)

    generate_dot(workflow, outfile)
    generate_viz(outfile)

    logfile.close()


if __name__ == '__main__':
    main()
