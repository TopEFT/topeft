# Library of functions for printing a latex table
# Assumes the dictionary you pass is of this format:
#   dict = {
#       k : {
#           subk : (val,err)
#       }
#   }

# Print a latex table for the yields
def print_latex_yield_table(yld_dict,proc_order_lst,cat_order_lst,tag,print_begin_info=False,print_end_info=False,print_errs=False,column_variable="cats"):

    def format_header(column_lst):
        s = "\\hline "
        for i,col in enumerate(column_lst):
            col = col.replace("_"," ")
            col = col.replace("cat","")
            s = s + " & " + col
        s = s + " \\\\ \\hline"
        return s

    def print_begin():
        print("\n")
        print("\\documentclass[10pt]{article}")
        print("\\usepackage[margin=0.05in]{geometry}")
        print("\\begin{document}")

    def print_end():
        print("\\end{document}")
        print("\n")

    def print_table(proc_lst,cat_lst,columns):
        print("\\begin{table}[hbtp!]")
        print("\\setlength\\tabcolsep{5pt}")
        print(f"\\caption{{{tag}}}") # Need to escape the "{" with another "{"
        print("\\smallskip")

        # Print categories as columns
        if columns == "cats":
            tabular_info = "c"*(len(cat_lst)+1)
            print(f"\\begin{{tabular}}{{{tabular_info}}}")
            print(format_header(cat_lst))
            for proc in proc_lst:
                if proc not in yld_dict.keys():
                    print("\\rule{0pt}{3ex} ","-",end=' ')
                    for cat in cat_lst:
                        print("&","-",end=' ')
                    print("\\\ ")
                else:
                    print("\\rule{0pt}{3ex} ",proc.replace("_"," "),end=' ')
                    for cat in cat_lst:
                        yld , err = yld_dict[proc][cat]
                        if yld is not None: yld = round(yld,2)
                        if err is not None: err = round(err,2)
                        if print_errs:
                            print("&",yld,"$\pm$",err,end=' ')
                        else:
                            print("&",yld,end=' ')
                print("\\\ ")

        # Print processes as columns
        if columns == "procs":
            tabular_info = "c"*(len(proc_lst)+1)
            print(f"\\begin{{tabular}}{{{tabular_info}}}")
            print(format_header(proc_lst))
            for cat in cat_lst:
                print("\\rule{0pt}{3ex} ",cat.replace("_"," "),end=' ')
                for proc in proc_lst:
                    if proc not in yld_dict.keys():
                        print("& - ",end=' ')
                    else:
                        yld , err = yld_dict[proc][cat]
                        if yld is not None: yld = round(yld,2)
                        if err is not None: err = round(err,2)
                        if print_errs:
                            print("&",yld,"$\pm$",err,end=' ')
                        else:
                            print("&",yld,end=' ')
                print("\\\ ")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    if print_begin_info: print_begin()
    print_table(proc_order_lst,cat_order_lst,columns=column_variable) # TODO: Need to fix this
    if print_end_info: print_end()

