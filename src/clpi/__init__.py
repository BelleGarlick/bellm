from typing import Dict, Union, TypeAlias, List, Type

from pydantic._internal._model_construction import ModelMetaclass

from clpi.base_clpi_model import BaseClpIModel

ParserMap: TypeAlias = Union[BaseClpIModel, Dict[str, "ITEMS"]]


def parse_vars(args: List[str]) -> Dict:
    arg_options = {}

    for arg in args:
        arg_key, arg_value = arg.split('=') if "=" in arg else (arg, "true")

        while arg_key.startswith("-"):
            arg_key = arg_key[1:]

        arg_options[arg_key] = arg_value

    return arg_options


def __build_and_run_function[T](args1, args2: Dict[str, str], config: Type[T]) -> T:
    creation_dict = {}

    # todo load file / predefined configs here

    for key, value in args2.items():
        current_dict_level = creation_dict

        toks = key.split('.')
        for idx, nested_key in enumerate(toks):
            if idx < len(toks) - 1:
                if nested_key not in current_dict_level:
                    current_dict_level[nested_key] = {}
                current_dict_level = current_dict_level[nested_key]

            else:
                current_dict_level[nested_key] = value

    # if self.save_config_path:
    #     print("saving config")
    #     with open(self.save_config_path, "w") as f:
    #         json.dump(creation_dict, f)

    # todo save configs here

    # todo could validate configs here too

    created_object = config(**creation_dict)

    # for key, field in config.model_fields.items():
    #     if isinstance(field.annotation, ModelMetaclass):
    #         breakpoint()
    #     breakpoint()

    # todo list config outputs here

    created_object.run(*args1, **args2)


def print_help(config: Type[BaseClpIModel], prefix=" --"):
    for field, value in config.model_fields.items():
        if isinstance(value.annotation, ModelMetaclass):
            print_help(value.annotation, prefix=prefix + f"{field}.")
        else:
            text = f"{prefix}{field}: {value.description or "No description provided"}"
            if value.default is not None:
                text = f"{text} (default: {value.default})"
            print(text)


def parse(args: List[str], parser_map: ParserMap):
    vars = parse_vars([x for x in args if x[0] == "-"])
    args = [x for x in args if not x.startswith('-')]

    tok_index = 0
    func_map = parser_map
    while True:
        if type(func_map) == ModelMetaclass:
            if "h" in vars or "help" in vars:
                print("-" * 60)
                print("Vars")
                print_help(func_map)

            else:
                __build_and_run_function(args, vars, func_map)

            break

        tok = args[tok_index] if tok_index < len(args) else None


        if tok is not None and tok in func_map:
            func_map = func_map[tok]

        elif tok is None and "__default" in func_map:
            func_map = func_map["__default"]

        else:
            print("Invalid command. Use `-h` or `--help` to view available options.")

            break

        tok_index += 1
