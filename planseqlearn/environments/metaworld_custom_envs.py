from collections import OrderedDict

import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
    SawyerHammerEnvV2,
    SawyerHandlePullEnvV2,
    SawyerNutAssemblyEnvV2,
)

MT3_CUSTOMIZED = OrderedDict(
    (
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
    ),
)

MT3_CUSTOMIZED_ARGS_KWARGS = {
    key: dict(
        args=[],
        kwargs={"task_id": list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)},
    )
    for key, _ in MT3_CUSTOMIZED.items()
}


class MT3_Customized(metaworld.Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT3_CUSTOMIZED
        self._test_classes = OrderedDict()
        train_kwargs = MT3_CUSTOMIZED_ARGS_KWARGS
        self._train_tasks = metaworld._make_tasks(
            self._train_classes, train_kwargs, metaworld._MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []


MT5_CUSTOMIZED = OrderedDict(
    (
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("handle-pull-v2", SawyerHandlePullEnvV2),
    ),
)

MT5_CUSTOMIZED_ARGS_KWARGS = {
    key: dict(
        args=[],
        kwargs={"task_id": list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)},
    )
    for key, _ in MT5_CUSTOMIZED.items()
}


class MT5_Customized(metaworld.Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT5_CUSTOMIZED
        self._test_classes = OrderedDict()
        train_kwargs = MT5_CUSTOMIZED_ARGS_KWARGS
        self._train_tasks = metaworld._make_tasks(
            self._train_classes, train_kwargs, metaworld._MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []


MT10_CUSTOMIZED = OrderedDict(
    (
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("door-unlock-v2", SawyerDoorUnlockEnvV2),
        ("door-lock-v2", SawyerDoorLockEnvV2),
        ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("handle-pull-v2", SawyerHandlePullEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
        ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
    ),
)

MT10_CUSTOMIZED_ARGS_KWARGS = {
    key: dict(
        args=[],
        kwargs={"task_id": list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)},
    )
    for key, _ in MT10_CUSTOMIZED.items()
}


class MT10_Customized(metaworld.Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT10_CUSTOMIZED
        self._test_classes = OrderedDict()
        train_kwargs = MT10_CUSTOMIZED_ARGS_KWARGS
        self._train_tasks = metaworld._make_tasks(
            self._train_classes, train_kwargs, metaworld._MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []


MT_DOOR = OrderedDict(
    (
        ("door-open-v2", SawyerDoorEnvV2),
        ("door-unlock-v2", SawyerDoorUnlockEnvV2),
        ("door-lock-v2", SawyerDoorLockEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
    ),
)

MT_DOOR_ARGS_KWARGS = {
    key: dict(
        args=[],
        kwargs={"task_id": list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)},
    )
    for key, _ in MT_DOOR.items()
}


class MT_Door(metaworld.Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT_DOOR
        self._test_classes = OrderedDict()
        train_kwargs = MT_DOOR_ARGS_KWARGS
        self._train_tasks = metaworld._make_tasks(
            self._train_classes, train_kwargs, metaworld._MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []
