import random
from utils import get_available_col_nums

def get_move(positions):
    available_col_nums = get_available_col_nums(positions)
    return random.choice(available_col_nums)
