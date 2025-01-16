import yaml
import numpy as np


def exec_try_run(config):
    for i in range(len(config)):
        with open(config[i]) as file:
            print(config[i])
            content = yaml.load(file, Loader=yaml.FullLoader)
            try:
                np.random.seed(content["params"]["seed"])
            except:
                print("No seed selected")
            for key, make in content["make"].items():
                if make:
                    eval(f"{key}(content)")
                try:
                    if make:
                        print(f"{key}(content)")

                except:
                    print(f"Could not find:{key}(content)")
