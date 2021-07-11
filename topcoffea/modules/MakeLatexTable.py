# Library of functions for printing a latex table
# Assumes the dictionary you pass is of this format:
#   vals_dict = {
#       key : {
#           subkey : (val,err)
#       }
#   }

# Make the header row
def format_header(column_lst):
    s = "\\hline "
    for i,col in enumerate(column_lst):
        col = col.replace("_"," ")
        col = col.replace("cat","")
        s = s + " & " + col
    s = s + " \\\\ \\hline"
    return s

# Print the info for the beginning of a latex document
def print_begin():
    print("\n")
    print("\\documentclass[10pt]{article}")
    print("\\usepackage[margin=0.05in]{geometry}")
    print("\\begin{document}")

# Print the info for the end of a latex document
def print_end():
    print("\\end{document}")
    print("\n")

# Print the body of the latex table
def print_table(vals_dict,key_order,subkey_order,caption_text,print_errs,columns):
    print("\\begin{table}[hbtp!]")
    print("\\setlength\\tabcolsep{5pt}")
    print(f"\\caption{{{caption_text}}}") # Need to escape the "{" with another "{"
    print("\\smallskip")

    # Print subkeys as columns
    if columns == "subkeys":
        tabular_info = "c"*(len(subkey_order)+1)
        print(f"\\begin{{tabular}}{{{tabular_info}}}")
        print(format_header(subkey_order))
        for key in key_order:
            if key not in vals_dict.keys():
                print("\\rule{0pt}{3ex} ","-",end=' ')
                for subkey in subkey_order:
                    print("&","-",end=' ')
                print("\\\ ")
            else:
                print("\\rule{0pt}{3ex} ",key.replace("_"," "),end=' ')
                for subkey in subkey_order:
                    val , err = vals_dict[key][subkey]
                    if val is not None: val = round(val,2)
                    if err is not None: err = round(err,2)
                    if print_errs:
                        print("&",val,"$\pm$",err,end=' ')
                    else:
                        print("&",val,end=' ')
            print("\\\ ")

    # Print keys as columns
    elif columns == "keys":
        tabular_info = "c"*(len(key_order)+1)
        print(f"\\begin{{tabular}}{{{tabular_info}}}")
        print(format_header(key_order))
        for subkey in subkey_order:
            print("\\rule{0pt}{3ex} ",subkey.replace("_"," "),end=' ')
            for key in key_order:
                if key not in vals_dict.keys():
                    print("& - ",end=' ')
                else:
                    val , err = vals_dict[key][subkey]
                    if val is not None: val = round(val,2)
                    if err is not None: err = round(err,2)
                    if print_errs:
                        print("&",val,"$\pm$",err,end=' ')
                    else:
                        print("&",val,end=' ')
            print("\\\ ")

    else:
        raise Exception(f"\nError: Unknown column type \"{columns}\". Exiting...\n")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

# Wrapper function for printing a table
def print_latex_yield_table(vals_dict,key_order_lst,cat_order_lst,tag,print_begin_info=False,print_end_info=False,print_errs=False,column_variable="subkeys"):
    if print_begin_info: print_begin()
    print_table(vals_dict,key_order_lst,cat_order_lst,tag,print_errs,columns=column_variable)
    if print_end_info: print_end()

