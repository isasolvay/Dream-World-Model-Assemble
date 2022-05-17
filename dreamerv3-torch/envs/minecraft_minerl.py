
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


def edit_options(**kwargs):
    import os, pathlib, re

    for word in os.popen("pip3 --version").read().split(" "):
        if "-packages/pip" in word:
            break
    else:
        raise RuntimeError("Could not found python package directory.")
    packages = pathlib.Path(word).parent
    filename = packages / "minerl/Malmo/Minecraft/run/options.txt"
    options = filename.read_text()
    if "fovEffectScale:" not in options:
        options += "fovEffectScale:1.0\n"
    if "simulationDistance:" not in options:
        options += "simulationDistance:12\n"
    for key, value in kwargs.items():
        assert f"{key}:" in options, key
        assert isinstance(value, str), (value, type(value))
        options = re.sub(f"{key}:.*\n", f"{key}:{value}\n", options)
    filename.write_text(options)


edit_options(
    difficulty="2",
    renderDistance="6",
    simulationDistance="6",
    fovEffectScale="0.0",
    ao="1",
    gamma="5.0",
)


class MineRLEnv(EnvSpec):
    def __init__(self, resolution=(64, 64), break_speed=50, gamma=10.0):
        self.resolution = resolution
        self.break_speed = break_speed
        self.gamma = gamma
        super().__init__(name="MineRLEnv-v1")

    def create_agent_start(self):
        return [
            BreakSpeedMultiplier(self.break_speed),
        ]

    def create_agent_handlers(self):
        return []

    def create_server_world_generators(self):
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self):
        return [handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_initial_conditions(self):
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=True,
                start_time=0,
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True,
            ),
        ]

    def create_observables(self):
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(mc.ALL_ITEMS),
            handlers.EquippedItemObservation(