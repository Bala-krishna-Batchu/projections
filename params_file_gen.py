'''Generate parameters file from provided template.'''

def arguments():
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-p', '--params-output',
        help='''Path to paramter file output locations (one for each JSON object). Alternatively, if no paths are supplied parameters will be printed to the screen only.''',
        type=str,
        nargs='*'
    )
    parser.add_argument(
        '-t', '--template',
        help='Path to parameters template',
        type=str,
        default='PBM_Opt_Code/CPMO_parameters_TEMPLATE.py'
    )
    parser.add_argument(
        '-c', '--custom-args-json',
        help='Path to JSON file with parameters in JSON formated object ({"client_name":"Sample", ...}.',
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Print parameters to stdout as well',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    return args

def get_custom_params(params_json):
    import json
    params_list = json.load(open(params_json))
    for params_dict in params_list:
        assert type(params_dict) == dict, 'Provided JSON object could not be converted to python dictionary'
    return params_list

def param_gen(template, custom_params, output_loc):
    '''Given custom_params dictionary, write parameter file to output_loc according to template.'''
    import jinja2 as jj2
    params_list = []
    if not output_loc:
            output_loc = ['']*len(custom_params)
    for cp, loc in zip(custom_params, output_loc):
        param_temp = jj2.Template(open(template).read())
        params = param_temp.render(**cp)
        if loc:
            with open(loc, 'w') as f:
                f.write(params)
        params_list.append(params)
    return params_list

if __name__ == '__main__':
    args = arguments()
    if args.custom_args_json:
        cparams = get_custom_params(args.custom_args_json)
    else:
        cparams = {}
    gparams = param_gen(args.template, cparams, args.params_output)
    if args.verbose:
        for p in gparams:
            print(gparams, end='\n\n')
    