import os
import time

import numpy
import numpy as np
import json
import pandas as pd

ROOT_FOLDER = "/home/kinovaresearch/Desktop/USER_STUDY_RESULTS"
TASKS = [
    "Reaching",
    "PickAndPlace",
    "Shelving"
]
TREATMENTS = {
    "A" : "Direct Teleoperation",
    "B" : "Shared Autonomy Baseline",
    "C" : "Vision Shared Autonomy (Ours)"
}

class UserStudyExperiment:
    def __init__(self):
        user_dirs = self.get_user_dirs()
        self.user_id = max(user_dirs) + 1 if len(user_dirs) > 0 else 0
        self.dir = os.path.join(ROOT_FOLDER, str(self.user_id))
        os.mkdir(self.dir)
        self.treatment_order = self.assign_treatment_order()
        self.print_treatments()
        self.create_folder_tree()
        self.set_active_user()

    def assign_treatment_order(self):
        experiments = {}
        # Add Familiarity Task
        for task in TASKS:
            experiments[task] = []
            treatments = list(TREATMENTS.keys())
            if task != "Reaching":
                np.random.shuffle(treatments)
            for i, treatment in enumerate(treatments):
                experiments[task].append(f"{i + 1}. {TREATMENTS[treatment]} ({treatment})")

        with open(os.path.join(self.dir, "treatments.json"), "w") as f:
            json.dump(experiments, f)

        return experiments

    def print_treatments(self):
        print(f"USER {self.user_id} ASSIGNED TO RANDOMIZED TREATMENT ORDER")
        for k in self.treatment_order:
            print(f"\n{k}")
            for t in self.treatment_order[k]:
                print(f"\t{t}")

    def create_folder_tree(self):
        for task in TASKS:
            task_dir = os.path.join(self.dir, task)
            os.mkdir(task_dir)
            for treatment in TREATMENTS:
                treatment_dir = os.path.join(task_dir, treatment)
                os.mkdir(treatment_dir)

    def set_active_user(self):
        np.savetxt(os.path.join(ROOT_FOLDER, "curr_user.txt"), np.array([self.user_id], dtype=np.int8), fmt="%5u")

    @staticmethod
    def get_active_user():
        user = np.loadtxt(os.path.join(ROOT_FOLDER, "curr_user.txt"), dtype=np.int8)
        return user

    @staticmethod
    def record_result(task, mode, task_duration, input_magnitude, t_f=time.time()):
        mode_translate = {0: "A", 1: "B", 2: "C"}

        assert task in TASKS
        assert mode in mode_translate

        user = UserStudyExperiment.get_active_user()
        dir = os.path.join(ROOT_FOLDER, str(user), task, mode_translate[mode], str(int(t_f)))
        os.mkdir(dir)

        columns = ["task_duration", "cumulative_input"]
        data = np.array([[task_duration, input_magnitude]])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(dir, f"results.csv"), index=False)

    @staticmethod
    def get_user_dirs():
        """
        Retrieve All Folder IDS from the ROOT_FOLDER containing a user experiment
        """
        dirs = []
        for d in os.listdir(ROOT_FOLDER):
            try:
                name = int(d)
            except Exception as e:
                continue
            dirs.append(name)
        return dirs


if __name__ == "__main__":
    study = UserStudyExperiment()
    active_user = UserStudyExperiment.get_active_user()
    print(f"PERSISTENT EXPERIMENT USER SET TO: {active_user}")