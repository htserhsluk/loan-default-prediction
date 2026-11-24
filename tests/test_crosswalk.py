import pandas as pd
from archive import verified_positions, KEYS

def test_collisions_and_repeat_members_are_removed():
    def row(n):
        return {key: str(n) for key in KEYS}
    archive = pd.DataFrame([row(i) for i in [1, 2, 3, 4, 5, 6]]).assign(member_id=["a","b","b","c","d",None])
    local = pd.DataFrame([row(i) for i in [1, 2, 3, 4, 4, 6]])
    assert verified_positions(archive, local).tolist() == [0]
