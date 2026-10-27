"""v2 physical parameters; only crowded-scene walls are removed."""
from experiment import *
from dataclasses import replace
from legacy_states import LiteratureConfig,rect
from legacy_states import config as original_config,walls as original_walls
import numpy as np

def config(scene,n=None):
    c=original_config(scene,n)
    if scene=='lit_crowded_random':c=replace(c,max_steps=int(50/.15))
    return c

def walls(scene):
    return np.zeros((0,2,2)) if scene=='lit_crowded_random' else original_walls(scene)
