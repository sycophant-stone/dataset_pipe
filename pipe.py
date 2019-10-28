import json
import ast
import re
import traceback
import sys
import time
import os
import argparse
import importlib
import yaml
import utils.shell as shell
import utils.json_encoder as json_encoder
import utils.filesystem as filesystem
from jinja2 import Template
import utils.concat as yaml_tags_concat

loaded_modnames = {}

def store_config_to_template_vars(step, config_file):
    store_config = yaml.load(filesystem.file_get_contents(config_file), Loader=yaml.Loader)
    template_vars = {}
    if step not in store_config:
        return template_vars
    for var in store_config[step]:
        template_vars[var] = store_config[step][var]
    return template_vars


def run_when(module):
    if 'when' not in module:
        return True
    if 'cwd' in module and module['cwd'] != '':
        os.chdir(module['cwd'])

    expression = module['when']
    m = importlib.import_module('fdp.utils.when').__dict__
    g = {}
    for key in m:
        if key.startswith('__'):
            continue
        g[key] = m[key]
    if type(expression) != bool:
        res = eval(expression, g)
        if type(res) is not bool:
            raise Exception('expected when(%s) ret bool' % expression)
        return res
    else:
        return expression


def run_when_store_match(module, store_name):
    if 'run_when_store_match' not in module:
        return True
    stores = module['run_when_store_match'].split(',')
    for s in stores:
        if store_name.find(s) >= 0:
            return True
    return False


def load_module(fn):
    fn_split = fn.split('.')
    # modname = 'ssd.hnren.Data.dataset_pipe.' + '.'.join(fn_split[:-1])
    modname = '.'.join(fn_split[:-1])
    # print("fn_split:%s, modname:%s"%(fn_split, modname))
    if modname not in loaded_modnames:
        loaded_modnames[modname] = True
        globals()[modname.replace('.', '_')] = importlib.import_module(modname)
    return getattr(globals()[modname.replace('.', '_')], fn_split[-1])


def call(cwd, fn, input, output):
    if cwd != '':
        print("call's cwd:%s"%(cwd))
        os.chdir(cwd)
    input.update(output)
    return load_module(fn)(**input)

def file_get_contents(filename):
    with open(filename, 'rb') as f:
        return f.read(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, help='INPUT: process define yaml', required=True)
    parser.add_argument('--data_dir', type=str, help='INPUT: data dir path', required=True)
    parser.add_argument('--baseline_dataset_dir', type=str, help='INPUT: baseline dataset dir', required=True)
    parser.add_argument('--caffe_python_dir', type=str, help='INPUT: caffe python dir ', required=True)
    parser.add_argument('--end_step', type=str, help='INPUT: whether pre stop at the end_step.', required=False)

    args = parser.parse_args()
    content = file_get_contents(args.yaml)
    template = Template(content.decode('utf-8'))

    pipeline_name = os.path.basename(args.yaml)[:-5]
    pipeline_mark_path = '/.pipeline.%s' % pipeline_name
    mark_finished_step = []
    if not os.path.isfile(pipeline_mark_path):
        mark_finished_step_file = open(pipeline_mark_path, mode='a')
    else:
        mark_finished_step = filesystem.file_get_content_as_lines(pipeline_mark_path)
        mark_finished_step_file = open(pipeline_mark_path, mode='a')
    print("arg.data_dir: ",args.data_dir)
    vars = {
        'data_dir': args.data_dir,
        'baseline_dataset_dir': args.baseline_dataset_dir,
        'caffe_python_dir': args.caffe_python_dir
    }
    yaml.Loader.add_constructor('!CONCAT', yaml_tags_concat.ConcatTag.from_yaml)
    yaml.Dumper.add_multi_representer(yaml_tags_concat.ConcatTag, yaml_tags_concat.ConcatTag.to_yaml)

    yaml_definition = template.render(vars)
    process_definition = yaml.load(yaml_definition, Loader=yaml.Loader)

    # execute module step by step
    step = 0
    for step_fn in process_definition:
        for step_fn_name in step_fn:
            module_name = shell.colored('step[%d]: %s' % (step, step_fn_name), 'cyan')
            mark_id = '%s[%d]' % (step_fn_name, step)
            if mark_id in mark_finished_step:
                print('finish %s', module_name)
                if step == args.end_step:
                    print('stop')
                    break
                continue
            print(shell.colored('step[%d]: === %s ===' % (step, step_fn[step_fn_name]['desc']), 'cyan'))
            display_info = []
            display_info.append(
                '%s\ncwd: %s' % (module_name, shell.colored(step_fn[step_fn_name].get('cwd', os.getcwd()), 'green')))
            if 'when' in step_fn[step_fn_name]:
                display_info.append('when: %s' % shell.colored(step_fn[step_fn_name]['when'], 'green'))
            display_info.append('input:\n%s' % shell.colored_json(json.dumps(step_fn[step_fn_name]['input'],
                                                                             cls=json_encoder.JsonEncoder, indent=4)))
            if 'output' in step_fn[step_fn_name]:
                display_info.append('output:\n%s' % shell.colored_json(json.dumps(step_fn[step_fn_name].get('output'),
                                                                                  cls=json_encoder.JsonEncoder,
                                                                                  indent=4)))
            if 'evaluate' in step_fn[step_fn_name]:
                display_info.append(
                    'evaluate:\n%s' % shell.colored_json(json.dumps(step_fn[step_fn_name].get('evaluate'),
                                                                    cls=json_encoder.JsonEncoder, indent=4)))
            print('\n'.join(display_info))

            cwd = step_fn[step_fn_name].get('cwd', '')
            call(cwd, step_fn_name, step_fn[step_fn_name]['input'],
                 step_fn[step_fn_name].get('output', {}))
            mark_finished_step_file.write('%s\n' % mark_id)
            step = step + 1

    mark_finished_step_file.close()

