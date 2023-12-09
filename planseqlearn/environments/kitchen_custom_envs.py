from d4rl.kitchen.kitchen_envs import KitchenBase
from gym.envs.registration import register

register(
    id="kitchen-microwave-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenMicrowaveV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-hinge-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenHingeCabinetV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-tlb-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenTopLeftBurnerV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-kettle-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenKettleV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-light-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenLightV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-slider-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenSliderV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-kettle-light-burner-slider-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenKettleLightBurnerSliderV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-kettle-light-burner-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenKettleLightBurnerV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-kettle-burner-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenKettleBurnerV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-light-burner-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenLightBurnerV0",
    max_episode_steps=280,
    kwargs={},
)

register(
    id="kitchen-microwave-kettle-light-burner-slider-v0",
    entry_point="planseqlearn.environments.kitchen_custom_envs:KitchenMicrowaveKettleLightBurnerSliderV0",
    max_episode_steps=280,
    kwargs={},
)


class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ["microwave"]


class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ["kettle"]


class KitchenLightV0(KitchenBase):
    TASK_ELEMENTS = ["light switch"]


class KitchenSliderV0(KitchenBase):
    TASK_ELEMENTS = ["slide cabinet"]


class KitchenHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["hinge cabinet"]


class KitchenTopLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top burner"]


class KitchenKettleLightBurnerSliderV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "light switch", "top burner", "slide cabinet"]


class KitchenKettleLightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "light switch", "top burner"]


class KitchenKettleBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "top burner"]


class KitchenLightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["light switch", "top burner"]


class KitchenMicrowaveKettleLightBurnerSliderV0(KitchenBase):
    TASK_ELEMENTS = [
        "microwave",
        "kettle",
        "light switch",
        "top burner",
        "slide cabinet",
    ]
